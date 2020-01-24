import numpy as np
from ReadAtmost import Atmost

def compute_cmass(atmost: Atmost) -> np.ndarray:
    cmass = np.zeros_like(atmost.z1)
    cmass[:, 0] = 1e-9
    for i in range(1, cmass.shape[1]):
        cmass[:, i] = cmass[:, i-1] + 0.5 * np.abs(atmost.z1[:, i-1] - atmost.z1[:, i]) * (atmost.d1[:, i] + atmost.d1[:, i-1])

    return cmass

def interp_to_const_cmass_grid(atmost: Atmost, cmass: np.ndarray, newGrid: np.ndarray) -> Atmost:
    z1 = np.zeros_like(atmost.z1)
    d1 = np.zeros_like(atmost.d1)
    ne1 = np.zeros_like(atmost.ne1)
    tg1 = np.zeros_like(atmost.tg1)
    vz1 = np.zeros_like(atmost.vz1)
    nh1 = np.zeros_like(atmost.nh1)

    for i in range(z1.shape[0]):
        z1[i] = np.interp(newGrid, cmass[i], atmost.z1[i])
        d1[i] = np.interp(newGrid, cmass[i], atmost.d1[i])
        ne1[i] = np.interp(newGrid, cmass[i], atmost.ne1[i])
        tg1[i] = np.interp(newGrid, cmass[i], atmost.tg1[i])
        vz1[i] = np.interp(newGrid, cmass[i], atmost.vz1[i])
        for j in range(nh1.shape[1]):
            nh1[i,j] = np.interp(newGrid, cmass[i], atmost.nh1[i,j])

    return Atmost(atmost.grav, atmost.tau2, atmost.vturb, atmost.time, atmost.dt, z1, d1, ne1, tg1, vz1, nh1, cgs=atmost.cgs)