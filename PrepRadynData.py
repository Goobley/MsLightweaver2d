import numpy as np
from radynpy.cdf import LazyRadynData
import matplotlib.pyplot as plt
from Interp import compute_cmass, interp_to_const_cmass_grid
from ReadAtmost import read_atmost
from scipy.interpolate import interp1d
import pickle

radyn = LazyRadynData('radyn_out.cdf')

# First interpolate beam heating onto chosen uniform grid
# Then do the timestep interpolation to match the MS timestep

atmost = read_atmost()
atmost.to_SI()
cmass = compute_cmass(atmost)
# cmassGrid = cmass[-1]
cmassGrid = cmass[0]
# cmassGrid = np.interp(np.linspace(0, 1, 400), np.linspace(0, 1, cmass[0].shape[0]), cmass[0])

staticAtmost = interp_to_const_cmass_grid(atmost, cmass, cmassGrid)

interpBheat = (atmost.bheat1.shape[0] == 0)
if interpBheat:
    bHeats = []
    for cdfIdx, t in enumerate(radyn.time):
        idx = np.searchsorted(atmost.time, t)
        bHeats.append(np.interp(cmassGrid, cmass[idx], radyn.bheat1[cdfIdx]))
        
    bHeatArr = np.array(bHeats)
    bHeat1 = interp1d(radyn.time, bHeatArr.T)(atmost.time).T

radynData = staticAtmost.__dict__
if interpBheat:
    radynData['bheat1'] = bHeat1

radynData['cmass'] = cmass
radynData['cmassGrid'] = cmassGrid

with open('RadynData.pickle', 'wb') as pkl:
    pickle.dump(radynData, pkl)