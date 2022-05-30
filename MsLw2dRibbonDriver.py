import pickle
import numpy as np
import matplotlib.pyplot as plt
from lightweaver.rh_atoms import H_6_atom, C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, Fe_atom, FeI_atom, MgII_atom, N_atom, Na_atom, S_atom, CaII_atom, He_9_atom
import lightweaver as lw
from MsLightweaverAtoms import H_6, CaII, H_6_nasa, CaII_nasa
from pathlib import Path
import os.path as path
import time
from MsLightweaverUtil import test_timesteps_in_dir, optional_load_starting_context
from MsLightweaver2dPeriodicCentralRibbon import MsLw2dPeriodic
from ReadAtmost import read_atmost

OutputDir = 'F9_flat_3e6_10flare_330_91_HAR/'
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

Nz = 330
NcentralColumnsFromFlare = 10
MaxZ = 3e6
Nquad2d = 11

startingCtx2d = optional_load_starting_context(OutputDir, suffix='2d')
xAxis = np.linspace(0, 2000e3, 41)
xAxis = np.concatenate([np.linspace(0, 500e3, 41), # [0, 0.5] Mm @ 10 km
                        np.linspace(0.525e6, 1.0e6, 20), # (0.5, 1.0] Mm @ 25 km
                        np.linspace(1.05e6, 3.0e6, 40)]) # (1.0, 3.0] Mm @ 50 km
xAxis = np.concatenate([np.linspace(-4e6, -2e6, 11),
                        np.linspace(-2e6, -0.5e6, 16)[1:],
                        np.linspace(-0.5e6, 0.5e6, 41)[1:],
                        np.linspace(0.5e6, 2e6, 16)[1:],
                        np.linspace(2e6, 4e6, 11)[1:]])

# xAxis = np.concatenate(((0,), np.linspace(2e3, 998e3, 20), (1000e3,)))

start = time.time()
ms2d = MsLw2dPeriodic(OutputDir, atmost, Nz, xAxis,
                      AtomSet,
                      activeAtoms=['H', 'Ca'],
                      startingCtx=startingCtx2d,
                      conserveCharge=ConserveCharge,
                      NcentralColumnsFromFlare=NcentralColumnsFromFlare,
                      maxZ=MaxZ,
                      saveJ=False)

firstStep = 0
if firstStep == 0:
    ms2d.initial_stat_eq(Nscatter=5)
    ms2d.save_timestep_data()

if startingCtx2d is None:
    with open(OutputDir + 'StartingContext2d.pickle', 'wb') as pkl:
        pickle.dump(ms2d.ctx, pkl)

maxSteps = ms2d.atmost.time.shape[0] - 1
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
    ms2d.time_dep_step(popsTol=2e-3, altJTol=1e-3, Nsubsteps=100)
    # ms.ctx.clear_ng()
    ms2d.save_timestep_data()
    stepEnd = time.time()
    print('-------')
    print('Timestep %d done (%f s)' % ((i+1), ms2d.atmost.time[i+1]))
    print('Time taken for step %.2e s' % (stepEnd - stepStart))
    print('-------')
end = time.time()
print('Total time taken %.4e' % (end - start))
