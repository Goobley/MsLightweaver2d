import pickle
import numpy as np
import matplotlib.pyplot as plt
from lightweaver.rh_atoms import C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, Fe_atom, FeI_atom, MgII_atom, N_atom, Na_atom, S_atom
from lightweaver.atmosphere import Atmosphere, ScaleType
from lightweaver.atomic_set import RadiativeSet, SpeciesStateTable
from lightweaver.atomic_table import get_global_atomic_table
from lightweaver.molecule import MolecularTable
from lightweaver.LwCompiled import LwContext
from lightweaver.utils import InitialSolution, NgOptions
import lightweaver.constants as Const
from typing import List
from copy import deepcopy
from MsLightweaverAtoms import H_6, CaII, He_9
import os.path as path
import time
from notify_run import Notify
from HydroWeno.Simulation import Grid
from HydroWeno.BCs import zero_grad_bc
from HydroWeno.Weno import reconstruct_weno_nm_z
from MsLightweaverAdvector import MsLightweaverAdvector
from scipy.interpolate import interp1d

def simple_advection_bcs():
    lower_bc = zero_grad_bc('Lower')
    upper_bc = zero_grad_bc('Upper')
    def apply_bcs(grid, V):
        lower_bc(grid, V)
        upper_bc(grid, V)

    return apply_bcs

class MsLightweaverManager:

    def __init__(self, atmost, numInterfaces, startingCtx=None):
        self.atmost = atmost
        self.at = get_global_atomic_table()
        self.idx = 0
        self.numInterfaces = numInterfaces

        if startingCtx is not None:
            self.ctx = startingCtx
            args = startingCtx.arguments
            self.atmos = args['atmos']
            self.spect = args['spect']
            self.aSet = self.spect.radSet
            self.eqPops = args['eqPops']
        else:
            nHTot = atmost['d1'][0] / (self.at.weightPerH * Const.Amu)
            self.atmos = Atmosphere(scale=ScaleType.ColumnMass, depthScale=np.copy(atmost['cmassGrid']), temperature=np.copy(atmost['tg1'][0]), vlos=np.copy(atmost['vz1'][0]), vturb=np.copy(atmost['vturb']), ne=np.copy(atmost['ne1'][0]), nHTot=nHTot)
            self.atmos.convert_scales()
            self.atmos.quadrature(5)

            self.aSet = RadiativeSet([H_6(), CaII(), He_9(), C_atom(), O_atom(), Si_atom(), Fe_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
            self.aSet.set_active('H', 'He', 'Ca')

            self.spect = self.aSet.compute_wavelength_grid()

            self.mols = MolecularTable()
            self.eqPops = self.aSet.compute_eq_pops(self.mols, self.atmos)
            self.atmos.bHeat = np.copy(self.atmost['bheat1'][0])
            self.atmos.hPops = self.eqPops['H']
            self.ctx = LwContext(self.atmos, self.spect, self.eqPops, initSol=InitialSolution.EscapeProbability, ngOptions=NgOptions(Norder=2, Nperiod=7, Ndelay=25))

        self.atmos.bHeat = np.copy(self.atmost['bheat1'][0])
        self.atmos.hPops = self.eqPops['H']

        numPopRows = 0
        for atom in self.aSet.activeAtoms:
            numPopRows += self.eqPops[atom.name].shape[0]
        
        self.atmos.convert_scales()
        # Expand grid slightly with linear extrapolation, so cmass interfaces are preserved better for interpolation back
        staticHeightInterfaces = interp1d(np.linspace(0, 1, self.atmos.height.shape[0]), self.atmos.height[::-1], fill_value='extrapolate')(np.linspace(-0.01, 1.01, numInterfaces))
        self.staticGrid = Grid(staticHeightInterfaces, 4)
        self.adv = MsLightweaverAdvector(self.staticGrid, self.atmos.height, self.atmos.cmass, self.atmost, simple_advection_bcs(), numPopRows=numPopRows)
        # Make height and density grid self-consistent with hydro. Slight offsets appear due to cell-centres vs interfaces
        self.atmos.height[:] = self.adv.height_from_cmass(self.atmos.cmass)
        newRho = self.adv.rho_from_cmass(self.atmos.cmass)
        self.atmos.nHTot[:] = newRho / (self.at.weightPerH*Const.Amu)
        self.eqPops.update_lte_atoms_Hmin_pops(self.atmos)
        self.JPrev = np.copy(self.ctx.spect.J)

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

    def advect_pops(self):
        activePops = []
        for atom in self.aSet.activeAtoms:
            activePops.append(self.eqPops[atom.name])
        pops = np.concatenate(activePops, axis=0)

        self.adv.fill_from_pops(self.atmos.cmass, pops)
        self.adv.step()

        newPops = self.adv.interp_to_cmass(self.atmos.cmass)
        newHeight = self.adv.height_from_cmass(self.atmos.cmass)
        newRho = self.adv.rho_from_cmass(self.atmos.cmass)

        self.atmos.height[:] = newHeight
        self.atmos.nHTot[:] = newRho / (self.at.weightPerH*Const.Amu)

        takeFrom = 0
        for atom in self.aSet.activeAtoms:
            atomPop = self.eqPops[atom.name]
            nLevels = atomPop.shape[0]
            atomPop[:] = newPops[takeFrom:takeFrom+nLevels, :]
            takeFrom += nLevels

    def increment_step(self):
        self.advect_pops()
        self.idx += 1
        self.atmos.temperature[:] = self.atmost['tg1'][self.idx]
        self.atmos.vlos[:] = self.atmost['vz1'][self.idx]
        self.atmos.ne[:] = self.atmost['ne1'][self.idx]
        # Now done with the advection
        # self.atmos.nHTot[:] = self.nHTot[self.idx]
        self.atmos.bHeat[:] = self.atmost['bheat1'][self.idx]

        # self.atmos.convert_scales()
        # NOTE(cmo): Convert scales is slow due to recomputing the tau500 opacity (pure python EOS), we only really need the height at the moment. So let's just do that quickly here.
        # Now done with the advection
        # self.update_height()
        self.ctx.update_deps()

    def time_dep_step(self, nSubSteps=100, popsTol=1e-3, JTol=3e-3):
        dt = self.atmost['dt'][self.idx+1]

        prevState = None
        for sub in range(nSubSteps):
            self.JPrev[:] = self.ctx.spect.J

            dJ = self.ctx.formal_sol_gamma_matrices()
            delta, prevState = self.ctx.time_dep_update(dt, prevState)

            maxDjLoc = np.unravel_index(np.argmax(np.abs(self.JPrev - self.ctx.spect.J) / self.JPrev), self.JPrev.shape)
            print('dJ index: %s' % repr(maxDjLoc))

            if sub >  2 and ((delta < popsTol and dJ < JTol) or delta < 0.5*popsTol):
                return
        else:
            self.ctx.time_dep_restore_prev_pops(prevState)
            raise ConvergenceError('Heck')

with open('RadynData.pickle', 'rb') as pkl:
    atmost = pickle.load(pkl)

if path.isfile('StartingContext.pickle'):
    with open('StartingContext.pickle', 'rb') as pkl:
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
    with open('Timesteps/Step_%.6d.pickle' % i, 'wb') as pkl:
        eqPops = distill_pops(deepcopy(ms.eqPops))
        Iwave = deepcopy(ms.ctx.spect.I)

        pickle.dump({'eqPops': eqPops, 'Iwave': Iwave}, pkl)

numInterfaces = 2001

start = time.time()
ms = MsLightweaverManager(atmost, numInterfaces, startingCtx=startingCtx)
ms.initial_stat_eq()

if startingCtx is None:
    with open('StartingContext.pickle', 'wb') as pkl:
        pickle.dump(ms.ctx, pkl)

save_timestep(0)

for i in range(ms.atmost['time'].shape[0] - 1):
    stepStart = time.time()
    if i != 0:
        ms.increment_step()
        ms.ctx.clear_ng()
    ms.time_dep_step(popsTol=1e-3, JTol=5e-3, nSubSteps=200)
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
