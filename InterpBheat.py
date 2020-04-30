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
# Finally interpolate onto radyn height grid for each timestep

atmost = read_atmost()
atmost.to_SI()

interpBheat = (atmost.bheat1.shape[0] == 0)

if interpBheat:
    Nspace = 50000
    height = np.copy(np.linspace(np.min(atmost.z1[:, -1]), np.max(atmost.z1[:, 0]), Nspace)[::-1])
    bHeats = []
    for cdfIdx, t in enumerate(radyn.time):
        idx = np.searchsorted(atmost.time, t)
        bHeats.append(interp1d(atmost.z1[idx], radyn.bheat1[cdfIdx], bounds_error=False, fill_value='extrapolate', kind='linear')(height))
        
    bHeatArr = np.array(bHeats)
    bheat1 = interp1d(radyn.time, bHeatArr.T, kind='linear')(atmost.time).T
    bheat1RadynGrid = np.zeros((bheat1.shape[0], atmost.z1.shape[1]))
    for i in range(bheat1.shape[0]):
        bheat1RadynGrid[i, :] = interp1d(height, bheat1[i], kind='linear')(atmost.z1[i])

    np.save('BheatInterp.npy', bheat1RadynGrid)