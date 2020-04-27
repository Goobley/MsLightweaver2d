import numpy as np
from radynpy.cdf import LazyRadynData
import matplotlib.pyplot as plt
from Interp import compute_cmass, interp_to_const_cmass_grid, interp_to_const_height_grid
from ReadAtmost import read_atmost
from scipy.interpolate import interp1d
import pickle

radyn = LazyRadynData('radyn_out.cdf')

# First interpolate beam heating onto chosen uniform grid
# Then do the timestep interpolation to match the MS timestep

atmost = read_atmost()
atmost.to_SI()
cmass = compute_cmass(atmost)
cmassGrid = cmass[-1]
# cmassGrid = cmass[0]
# cmassGrid = np.interp(np.linspace(0, 1, 400), np.linspace(0, 1, cmass[0].shape[0]), cmass[0])

NumPoints = 1000
ResolutionSplitPoint = 2e6
LowerFraction = 0.7
LowerPoints = int(NumPoints * LowerFraction)
UpperPoints = NumPoints - LowerPoints + 1

height = np.copy(np.concatenate([
            np.linspace(atmost.z1[0, -1], ResolutionSplitPoint, LowerPoints),
            np.linspace(ResolutionSplitPoint, atmost.z1[0, 0], UpperPoints)[1:]
                                ])[::-1])
Nspace = radyn.z1.shape[1]
height = interp1d(np.linspace(0, 1, Nspace), atmost.z1[0])(np.linspace(0, 1, NumPoints))
# height = atmost.z1[0]

HeightGrid = True


interp_fn = lambda pts, x, y: interp1d(x, y, kind=3)(pts)
if HeightGrid:
    staticAtmost = interp_to_const_height_grid(atmost, height, interp_fn=interp_fn)
else:
    staticAtmost = interp_to_const_cmass_grid(atmost, cmass, cmassGrid, interp_fn=interp_fn)

interpBheat = (atmost.bheat1.shape[0] == 0)
if interpBheat:
    bHeats = []
    for cdfIdx, t in enumerate(radyn.time):
        idx = np.searchsorted(atmost.time, t)
        bHeats.append(interp_fn(height, atmost.z1[idx], radyn.bheat1[cdfIdx]))
        
    bHeatArr = np.array(bHeats)
    bHeat1 = interp1d(radyn.time, bHeatArr.T)(atmost.time).T

radynData = staticAtmost.__dict__
if interpBheat:
    radynData['bheat1'] = bHeat1

if HeightGrid:
    radynData['zGrid'] = height
else:
    radynData['cmass'] = cmass
    radynData['cmassGrid'] = cmassGrid

with open('RadynData.pickle', 'wb') as pkl:
    pickle.dump(radynData, pkl)