import pickle
import numpy as np
import matplotlib.pyplot as plt
from lightweaver.rh_atoms import C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, Fe_atom, FeI_atom, MgII_atom, N_atom, Na_atom, S_atom
from lightweaver.atmosphere import Atmosphere, ScaleType
from lightweaver.atomic_set import RadiativeSet, SpeciesStateTable
from lightweaver.atomic_table import get_global_atomic_table
from lightweaver.molecule import MolecularTable
from lightweaver.LwCompiled import LwContext
from lightweaver.utils import InitialSolution
import lightweaver.constants as Const
from typing import List
from copy import deepcopy
from MsLightweaverAtoms import H_6, CaII, He_9
import os
import os.path as path
import time
from notify_run import Notify

OutputDir = 'TimestepsRadynAtomsCrossBurgessVdwCdi_NoBurgess/'

class MsLightweaverManager:

    def __init__(self, atmost, startingCtx=None):
        self.atmost = atmost
        self.at = get_global_atomic_table()
        self.nHTot = atmost['d1'] / (self.at.weightPerH*Const.Amu)
        self.idx = 0

        if startingCtx is not None:
            self.ctx = startingCtx
            args = startingCtx.arguments
            self.atmos = args['atmos']
            self.spect = args['spect']
            self.aSet = self.spect.radSet
            self.eqPops = args['eqPops']
        else:
            self.atmos = Atmosphere(scale=ScaleType.ColumnMass, depthScale=atmost['cmassGrid'], temperature=atmost['tg1'][0], vlos=atmost['vz1'][0], vturb=atmost['vturb'], ne=atmost['ne1'][0], nHTot=self.nHTot[0])
            self.atmos.convert_scales()
            self.atmos.quadrature(5)

            self.aSet = RadiativeSet([H_6(), CaII(), He_9(), C_atom(), O_atom(), Si_atom(), Fe_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
            self.aSet.set_active('H', 'Ca')

            self.spect = self.aSet.compute_wavelength_grid()

            self.mols = MolecularTable()
            self.eqPops = self.aSet.compute_eq_pops(self.mols, self.atmos)
            self.ctx = LwContext(self.atmos, self.spect, self.eqPops, initSol=InitialSolution.Lte)

        self.atmos.bHeat = self.atmost['bheat1'][0]
        self.atmos.hPops = self.eqPops['H']

    def initial_stat_eq(self, nScatter=3, nMaxIter=1000):
        for i in range(nMaxIter):
            dJ = self.ctx.formal_sol_gamma_matrices()
            if i < nScatter:
                continue

            delta = self.ctx.stat_equil()

            if self.ctx.crswDone and dJ < 3e-3 and delta < 1e-3:
                print('Stat eq converged in %d iterations' % (i+1))
                break

    def update_height(self):
        height = self.atmos.height
        cmass = self.atmos.cmass
        rhoSI = self.atmost['d1'][self.idx]
        height[0] = 0.0
        for k in range(1, cmass.shape[0]):
            height[k] = height[k-1] - 2.0 * (cmass[k] - cmass[k-1]) / (rhoSI[k-1] + rhoSI[k])

        # NOTE(cmo): ish.
        # NOTE(cmo): What I mean by ish, is this assumes that tau500 doesn't change, which it clearly does, but we never refer to tau500 in the computation or save it, so it currently doesn't matter.
        hTau1 = np.interp(1.0, self.atmos.tau_ref, height)
        height -= hTau1

    def increment_step(self):
        self.idx += 1
        self.atmos.temperature[:] = self.atmost['tg1'][self.idx]
        self.atmos.vlos[:] = self.atmost['vz1'][self.idx]
        self.atmos.ne[:] = self.atmost['ne1'][self.idx]
        self.atmos.nHTot[:] = self.nHTot[self.idx]
        self.atmos.bHeat[:] = self.atmost['bheat1'][self.idx]

        # self.atmos.convert_scales()
        # NOTE(cmo): Convert scales is slow due to recomputing the tau500 opacity (pure python EOS), we only really need the height at the moment. So let's just do that quickly here.
        self.update_height()
        self.ctx.update_deps()

    def time_dep_step(self, nSubSteps=100, popsTol=1e-3, JTol=3e-3):
        dt = self.atmost['dt'][self.idx+1]

        prevState = None
        for sub in range(nSubSteps):
            dJ = self.ctx.formal_sol_gamma_matrices()
            delta, prevState = self.ctx.time_dep_update(dt, prevState)

            if delta < popsTol and dJ < JTol:
                break

filesInOutDir = [f for f in os.listdir(OutputDir) if f.startswith('Step_')]
if len(filesInOutDir) > 0:
    print('Timesteps already present in output directory (%s), proceed? [Y/n]' % OutputDir)
    inp = input()
    if len(inp) > 0 and inp[0].lower() == 'n':
        raise ValueError('Data in output directory')

with open('RadynData.pickle', 'rb') as pkl:
    atmost = pickle.load(pkl)

if path.isfile(OutputDir + 'StartingContext.pickle'):
    with open(OutputDir + 'StartingContext.pickle', 'rb') as pkl:
        startingCtx = pickle.load(pkl)
else:
    startingCtx = None

def convert_atomic_pops(atom):
    d = {}
    if atom.pops is not None:
        d['n'] = atom.pops
    else:
        d['n'] = atom.pops
    d['nStar'] = atom.nStar
    return d

def distill_pops(eqPops):
    d = {}
    for atom in eqPops.atomicPops:
        d[atom.name] = convert_atomic_pops(atom)
    return d

def save_timestep(i):
    with open(OutputDir + 'Step_%.6d.pickle' % i, 'wb') as pkl:
        eqPops = distill_pops(ms.eqPops)
        Iwave = ms.ctx.spect.I
        pickle.dump({'eqPops': eqPops, 'Iwave': Iwave}, pkl)

start = time.time()
ms = MsLightweaverManager(atmost, startingCtx=startingCtx)
ms.initial_stat_eq()
save_timestep(0)


if startingCtx is None:
    with open(OutputDir + 'StartingContext.pickle', 'wb') as pkl:
        pickle.dump(ms.ctx, pkl)

for i in range(ms.atmost['time'].shape[0] - 1):
    stepStart = time.time()
    if i != 0:
        ms.increment_step()
    ms.time_dep_step(popsTol=1e-2, JTol=2e-2)
    save_timestep(i+1)
    stepEnd = time.time()
    print('-------')
    print('Timestep %d done (%f s)' % ((i+1), ms.atmost['time'][i+1]))
    print('Time taken for step %.2e s' % (stepEnd - stepStart))
    print('-------')
end = time.time()
print('Total time taken %.4e' % (end - start))

notify = Notify()
notify.read_config()
notify.send('MsLightweaver done!')
