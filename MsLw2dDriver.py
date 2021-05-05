import pickle
import numpy as np
import matplotlib.pyplot as plt
from lightweaver.rh_atoms import H_6_atom, C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, Fe_atom, FeI_atom, MgII_atom, N_atom, Na_atom, S_atom, CaII_atom, He_9_atom
import lightweaver as lw
from MsLightweaverAtoms import H_6, CaII, H_6_nasa, CaII_nasa
from pathlib import Path
import os
import os.path as path
import time
from MsLightweaverUtil import test_timesteps_in_dir, optional_load_starting_context
from MsLightweaver2dFixedIlluminationManager import MsLw2d
from ReadAtmost import read_atmost
from weno4 import weno4

OutputDir = 'F9_flat_450_40_nr_para_1stColCopy/'
Path(OutputDir).mkdir(parents=True, exist_ok=True)
NasaAtoms = [H_6_nasa(), CaII_nasa(), He_9_atom(), C_atom(), O_atom(), Si_atom(), Fe_atom(),
             MgII_atom(), N_atom(), Na_atom(), S_atom()]
FchromaAtoms = [H_6(), CaII(), He_9_atom(), C_atom(), O_atom(), Si_atom(), Fe_atom(),
                MgII_atom(), N_atom(), Na_atom(), S_atom()]
AtomSet = FchromaAtoms
ConserveCharge = True

atmost = read_atmost('atmost.dat')
atmost.to_SI()
if atmost.bheat1.shape[0] == 0:
    atmost.bheat1 = np.load('BheatInterp.npy')

with open('ZGrid2d_1037.pkl', 'rb') as pkl:
    zAxis = pickle.load(pkl)['zAxis']

startingCtx1d = optional_load_starting_context(OutputDir, suffix='1d')
startingCtx2d = optional_load_starting_context(OutputDir, suffix='2d')
xAxis = np.linspace(0, 2000e3, 40)
Nz = 450
# xAxis = np.concatenate(((0,), np.linspace(2e3, 998e3, 20), (1000e3,)))

start = time.time()
ms2d = MsLw2d(OutputDir, atmost, Nz, xAxis,
              AtomSet,
              activeAtoms=['H', 'Ca'],
              startingCtx1d=startingCtx1d, startingCtx=startingCtx2d,
              conserveCharge=ConserveCharge, saveJ=False, firstColumnFrom1d=True)

if startingCtx1d is None:
    with open(OutputDir + 'StartingContext1d.pickle', 'wb') as pkl:
        pickle.dump(ms2d.ms.ctx, pkl)

ms2d.initial_stat_eq()
ms2d.save_timestep_data()

if startingCtx2d is None:
    with open(OutputDir + 'StartingContext2d.pickle', 'wb') as pkl:
        pickle.dump(ms2d.ctx, pkl)

maxSteps = ms2d.atmost.time.shape[0] - 1
firstStep = 0
if firstStep != 0:
    # NOTE(cmo): This loads the state at the end of firstStep, therefore we
    # need to start integrating at firstStep+1
    ms2d.load_timestep(firstStep)
    # ms.ctx.spect.J[:] = 0.0
    ms2d.ctx.formal_sol_gamma_matrices()
    firstStep += 1

for i in range(firstStep, maxSteps):
    stepStart = time.time()
    if i != 0:
        ms2d.increment_step()
    ms2d.time_dep_step(popsTol=1e-3, Nsubsteps=1000)
    # ms.ctx.clear_ng()
    ms2d.save_timestep_data()
    stepEnd = time.time()
    print('-------')
    print('Timestep %d done (%f s)' % ((i+1), ms2d.atmost.time[i+1]))
    print('Time taken for step %.2e s' % (stepEnd - stepStart))
    print('-------')
end = time.time()
print('Total time taken %.4e' % (end - start))
