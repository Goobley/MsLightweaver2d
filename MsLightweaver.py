import pickle
import numpy as np
import matplotlib.pyplot as plt
from lightweaver.rh_atoms import H_6_atom, C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, Fe_atom, FeI_atom, MgII_atom, N_atom, Na_atom, S_atom, CaII_atom, He_9_atom
import lightweaver as lw
from MsLightweaverAtoms import H_6, CaII, H_6_nasa, CaII_nasa, H_6_nobb, H_6_noLybb, H_6_noLybbbf
from pathlib import Path
import os
import os.path as path
import time
# from notify_run import Notify
from MsLightweaverManager import MsLightweaverManager
from MsLightweaverUtil import test_timesteps_in_dir, optional_load_starting_context
from ReadAtmost import read_atmost
# from threadpoolctl import threadpool_limits
# threadpool_limits(1)
from RadynEmistab import EmisTable

OutputDir = 'TimestepsAllNoNonThermHNoLybb/'
Path(OutputDir).mkdir(parents=True, exist_ok=True)
Path(OutputDir + '/Rfs').mkdir(parents=True, exist_ok=True)
Path(OutputDir + '/ContFn').mkdir(parents=True, exist_ok=True)
NasaAtoms = [H_6_nasa(), CaII_nasa(), He_9_atom(), C_atom(), O_atom(), Si_atom(), Fe_atom(),
             MgII_atom(), N_atom(), Na_atom(), S_atom()]
FchromaAtoms = [H_6(), CaII(), He_9_atom(), C_atom(), O_atom(), Si_atom(), Fe_atom(),
                MgII_atom(), N_atom(), Na_atom(), S_atom()]
FchromaNoHbbAtoms = [H_6_nobb(), CaII(), He_9_atom(), C_atom(), O_atom(), Si_atom(), Fe_atom(),
                MgII_atom(), N_atom(), Na_atom(), S_atom()]
FchromaNoLybbAtoms = [H_6_noLybb(), CaII(), He_9_atom(), C_atom(), O_atom(), Si_atom(), Fe_atom(),
                MgII_atom(), N_atom(), Na_atom(), S_atom()]
FchromaNoLybbbfAtoms = [H_6_noLybbbf(), CaII(), He_9_atom(), C_atom(), O_atom(), Si_atom(), Fe_atom(),
                MgII_atom(), N_atom(), Na_atom(), S_atom()]
FchromaNoHbbNoContAtoms = [H_6_nobb(), CaII(), He_9_atom()]
AtomSet = FchromaNoLybbAtoms

# Removing Fang rates
del AtomSet[0].collisions[-1]
lw.atomic_model.reconfigure_atom(AtomSet[0])

ConserveCharge = False
PopulationTransportMode = 'Advect'
Prd = False
DetailedH = True
DetailedHPath = 'TimestepsAllNoNonThermH/'
# CoronalIrradiation = EmisTable('emistab.dat')
CoronalIrradiation = None
ActiveAtoms = ['H', 'Ca']

test_timesteps_in_dir(OutputDir)

atmost = read_atmost('atmost.dat')
atmost.to_SI()
if atmost.bheat1.shape[0] == 0:
    try:
        atmost.bheat1 = np.load('BheatInterp.npy')
    except:
        print('Unable to find BheatInterp.npy, press enter to continue without non-thermal beam rates')
        input()
        atmost.bheat1 = np.zeros_like(atmost.vz1)

startingCtx = optional_load_starting_context(OutputDir)

start = time.time()
if ConserveCharge and 'He' in ActiveAtoms:
    msFixedNe = MsLightweaverManager(atmost=atmost, outputDir=OutputDir,
                            atoms=AtomSet,
                            activeAtoms=ActiveAtoms, startingCtx=startingCtx,
                            detailedH=DetailedH,
                            detailedHPath=DetailedHPath,
                            conserveCharge=False,
                            populationTransportMode=PopulationTransportMode,
                            prd=Prd, downgoingRadiation=CoronalIrradiation)
    msFixedNe.initial_stat_eq(popTol=1e-3, Nscatter=20)
ms = MsLightweaverManager(atmost=atmost, outputDir=OutputDir,
                          atoms=AtomSet,
                          activeAtoms=ActiveAtoms, startingCtx=startingCtx,
                          detailedH=DetailedH,
                          detailedHPath=DetailedHPath,
                          conserveCharge=ConserveCharge,
                          populationTransportMode=PopulationTransportMode,
                          prd=Prd, downgoingRadiation=CoronalIrradiation)
if ConserveCharge and 'He' in ActiveAtoms:
    ms.ctx.eqPops['He'][...] = msFixedNe.ctx.eqPops['He']
ms.initial_stat_eq(popTol=1e-3, Nscatter=20)
ms.save_timestep()

# NOTE(cmo): Due to monkey-patching we can't reload the context currently
# if startingCtx is None:
#     with open(OutputDir + 'StartingContext.pickle', 'wb') as pkl:
#         pickle.dump(ms.ctx, pkl)

maxSteps = ms.atmost.time.shape[0] - 1
ms.atmos.bHeat[:] = ms.atmost.bheat1[0]
firstStep = 0
if firstStep != 0:
    # NOTE(cmo): This loads the state at the end of firstStep, therefore we
    # need to start integrating at firstStep+1
    ms.load_timestep(firstStep)
    ms.ctx.spect.J[:] = 0.0
    ms.ctx.formal_sol_gamma_matrices()
    firstStep += 1

for i in range(firstStep, maxSteps):
    stepStart = time.time()
    if i != 0:
        ms.increment_step()
    ms.time_dep_step(popsTol=1e-3, JTol=5e-3, nSubSteps=1000, theta=1.0)
    ms.ctx.clear_ng()
    ms.save_timestep()
    stepEnd = time.time()
    print('-------')
    print('Timestep %d done (%f s)' % ((i+1), ms.atmost.time[i+1]))
    print('Time taken for step %.2e s' % (stepEnd - stepStart))
    print('-------')
end = time.time()
print('Total time taken %.4e' % (end - start))

# notify = Notify()
# notify.read_config()
# notify.send('MsLightweaver done!')
