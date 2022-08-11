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
from MsLightweaverInterpManagerCentralFlare import MsLightweaverInterpManager
from ReadAtmost import read_atmost
from scipy.signal import wiener

def next_z_axis(atmost, Nz, idx, maxZ=None):
    idxO = 0
    idxN = idx
    DistTol = 1
    PointTotal = Nz
    SmoothingSize = 15
    HalfSmoothingSize = SmoothingSize // 2

    # Merge grids
    uniqueCombined = list(np.unique(np.sort(np.concatenate((atmost.z1[idxO],
                                                            atmost.z1[idxN])))))
    if maxZ is not None:
        zMaxIdx = np.searchsorted(uniqueCombined, maxZ) + 1
        uniqueCombined = uniqueCombined[:zMaxIdx]

    while True:
        diff = np.diff(uniqueCombined)
        if diff.min() > DistTol:
            break

        del uniqueCombined[diff.argmin() + 1]

    # Smooth
    uniqueCombined = np.sort(np.array(uniqueCombined))
    ucStart = uniqueCombined[0]
    uniqueCombined = wiener(uniqueCombined - ucStart, SmoothingSize) + ucStart

    # Fix ends
    uniqueCombined = list(uniqueCombined)
    del uniqueCombined[:HalfSmoothingSize]
    del uniqueCombined[-HalfSmoothingSize:]
    z1O = np.copy(atmost.z1[idxO][::-1])
    startIdx = np.searchsorted(z1O, uniqueCombined[0])
    endIdx = np.searchsorted(z1O, uniqueCombined[-1])
    maxZIdx = np.searchsorted(z1O, maxZ) + 1 if maxZ is not None else None
    for v in z1O[:startIdx]:
        uniqueCombined.append(v)
    for v in z1O[endIdx:maxZIdx]:
        uniqueCombined.append(v)
    uniqueCombined = np.sort(uniqueCombined)

    if uniqueCombined.shape[0] > PointTotal:
        # Remove every other point starting from top/bottom until we reach desired number of points
        UpperPointRemovalFraction = 3/4
        UpperPointsToRemove = int(UpperPointRemovalFraction * (uniqueCombined.shape[0] - PointTotal))

        # Is this efficient? No. But it should do for what we need right now
        upper = -HalfSmoothingSize
        for _ in range(UpperPointsToRemove):
            uniqueCombined = np.delete(uniqueCombined, upper)
            upper -= 1 # We only subtract 1 because the previous upper now points to the point below the one we just deleted

        LowerPointsToRemove = uniqueCombined.shape[0] - PointTotal
        lower = HalfSmoothingSize
        for _ in range(LowerPointsToRemove):
            uniqueCombined = np.delete(uniqueCombined, lower)
            lower += 1
    else:
        for i in range(PointTotal - uniqueCombined.shape[0]):
            a = np.diff(uniqueCombined).argmax()
            uniqueCombined = np.insert(uniqueCombined, a + 1,
                                        0.5 * (uniqueCombined[a] + uniqueCombined[a+1]))


    return np.copy(uniqueCombined[::-1])

OutputDir = 'F9_flat_3e6_5flare_330_91_HAR/'
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
NcentralColumnsFromFlare = 5
MaxZ = 3e6

startingCtx1d = optional_load_starting_context(OutputDir, suffix='1d')

start = time.time()
initialZGrid = next_z_axis(atmost, Nz, 1, MaxZ)
ms = MsLightweaverInterpManager(atmost, OutputDir, AtomSet, initialZGrid, 
                                conserveCharge=ConserveCharge)

ms.initial_stat_eq(Nscatter=5)
ms.save_timestep()

if startingCtx1d is None:
    with open(OutputDir + 'StartingContext1d.pickle', 'wb') as pkl:
        pickle.dump(ms.ctx, pkl)

maxSteps = ms.atmost.time.shape[0] - 1
firstStep = 0
if firstStep != 0:
    # NOTE(cmo): This loads the state at the end of firstStep, therefore we
    # need to start integrating at firstStep+1
    ms.load_timestep(firstStep)
    # ms.ctx.spect.J[:] = 0.0
    ms.ctx.formal_sol_gamma_matrices()
    firstStep += 1

for i in range(firstStep, maxSteps):
    stepStart = time.time()
    if i != 0:
        newZ = next_z_axis(atmost, Nz, ms.idx + 1, MaxZ)
        ms.increment_step(newZ)
    ms.time_dep_step(popsTol=1e-4)
    # ms.ctx.clear_ng()
    ms.save_timestep()
    stepEnd = time.time()
    print('-------')
    print('Timestep %d done (%f s)' % ((i+1), ms.atmost.time[i+1]))
    print('Time taken for step %.2e s' % (stepEnd - stepStart))
    print('-------')
end = time.time()
print('Total time taken %.4e' % (end - start))
