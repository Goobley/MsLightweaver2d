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
import os.path as path
import time

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
            self.aSet = self.spect.activeSet
            self.eqPops = args['eqPops']
        else:
            self.atmos = Atmosphere(scale=ScaleType.ColumnMass, depthScale=atmost['cmassGrid'], temperature=atmost['tg1'][0], vlos=atmost['vz1'][0], vturb=atmost['vturb'], ne=atmost['ne1'][0], nHTot=self.nHTot[0])
            self.atmos.convert_scales()
            self.atmos.quadrature(5)

            self.aSet = RadiativeSet([H_6(), CaII(), He_9(), C_atom(), O_atom(), Si_atom(), Fe_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
            self.aSet.set_active('H', 'He', 'Ca')

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

    def increment_step(self):
        self.idx += 1
        self.atmos.temperature[:] = self.atmost['tg1'][self.idx]
        self.atmos.vlos[:] = self.atmost['vz1'][self.idx]
        self.atmos.ne[:] = self.atmost['ne1'][self.idx]
        self.atmos.nHTot[:] = self.nHTot[self.idx]
        self.atmos.bHeat[:] = self.atmost['bheat1'][self.idx]

        # self.atmos.convert_scales()
        self.ctx.update_deps()

    def time_dep_step(self, nSubSteps=100, popsTol=1e-3, JTol=3e-3):
        dt = self.atmost['dt'][self.idx+1]

        prevState = None
        for sub in range(nSubSteps):
            dJ = self.ctx.formal_sol_gamma_matrices()
            delta, prevState = self.ctx.time_dep_update(dt, prevState)

            if delta < popsTol and dJ < JTol:
                break


with open('RadynData.pickle', 'rb') as pkl:
    atmost = pickle.load(pkl)

if path.isfile('StartingContext.pickle'):
    with open('StartingContext.pickle', 'rb') as pkl:
        startingCtx = pickle.load(pkl)
else:
    startingCtx = None

start = time.time()
ms = MsLightweaverManager(atmost, startingCtx=startingCtx)

eqPops : List[SpeciesStateTable] = []
ms.initial_stat_eq()
eqPops.append(deepcopy(eqPops))
for i in range(ms.atmost['time'].shape[0] - 1):
    stepStart = time.time()
    ms.increment_step()
    ms.time_dep_step(popsTol=1e-2, JTol=2e-2)
    eqPops.append(deepcopy(ms.eqPops))
    stepEnd = time.time()
    print('-------')
    print('Timestep %d done (%f)' % ((i+1), ms.atmost['time'][i+1]))
    print('Time taken for step %.2e' % (stepEnd - stepStart))
end = time.time()
print('Total time taken %.4e' % (end - start))