import pickle
import numpy as np
import matplotlib.pyplot as plt
from lightweaver.rh_atoms import H_6_atom, C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, Fe_atom, FeI_atom, MgII_atom, N_atom, Na_atom, S_atom, CaII_atom, He_9_atom
import lightweaver as lw
from MsLightweaverAtoms import H_6, CaII, H_6_nasa, H_6_nasa_new, CaII_nasa
from pathlib import Path
import os.path as path
import time
from MsLightweaverUtil import test_timesteps_in_dir, optional_load_starting_context
from MsLightweaver2dPeriodicCentralRibbonDouble import MsLw2dPeriodic
from ReadAtmost import read_atmost, read_cdf

OutputDir = 'F10_flat_3.5e6_5flare_370_91_7ray/'
Path(OutputDir).mkdir(parents=True, exist_ok=True)
NasaAtoms = [H_6_nasa_new(), CaII_nasa(), He_9_atom(), C_atom(), O_atom(), Si_atom(), Fe_atom(),
             MgII_atom(), N_atom(), Na_atom(), S_atom()]
FchromaAtoms = [H_6(), CaII(), He_9_atom(), C_atom(), O_atom(), Si_atom(), Fe_atom(),
                MgII_atom(), N_atom(), Na_atom(), S_atom()]
AtomSet = NasaAtoms
ConserveCharge = True

atmost = read_cdf('QS_10Mm_3MK_electron_longrampup_d5_Ec0025_RC.cdf')
atmost.to_SI()

Nz = 200 # Maybe.
# 300 km for each of these
NColumnsFromFlare1 = 300 // 25 # 12
NColumnsFromFlare2 = 300 // 25
Flare1StartIdx = 0
Flare2StartIdx = 600
MaxZ = 2e6

startingCtx2d = optional_load_starting_context(OutputDir, suffix='2d')
xAxis = np.concatenate([np.linspace(-4e6, -2e6, 11),
                        np.linspace(-2e6, -1e6, 11)[1:],
                        np.linspace(-1e6, 1e6, 81)[1:],
                        np.linspace(1e6, 2e6, 11)[1:],
                        np.linspace(2e6, 4e6, 11)[1:]])

# xAxis = np.concatenate(((0,), np.linspace(2e3, 998e3, 20), (1000e3,)))

start = time.time()
ms2d = MsLw2dPeriodic(OutputDir, atmost, Nz, xAxis,
                      AtomSet,
                      activeAtoms=['H'],
                      startingCtx=startingCtx2d,
                      conserveCharge=ConserveCharge,
                      NcolsFlare1=NColumnsFromFlare1,
                      NcolsFlare2=NColumnsFromFlare2,
                      flare1StartIdx=Flare1StartIdx,
                      flare2StartIdx=Flare2StartIdx,
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
