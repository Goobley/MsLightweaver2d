import numpy as np
from ReadAtmost import Atmost
import lightweaver as lw

def compute_cmass(atmost: Atmost) -> np.ndarray:
    cmass = np.zeros_like(atmost.z1)
    cmass[:, 0] = 1e-9
    for i in range(1, cmass.shape[1]):
        cmass[:, i] = cmass[:, i-1] + 0.5 * np.abs(atmost.z1[:, i-1] - atmost.z1[:, i]) * (atmost.d1[:, i] + atmost.d1[:, i-1])

    return cmass

def interp_to_const_cmass_grid(atmost: Atmost, cmass: np.ndarray, newGrid: np.ndarray, interp_fn=np.interp) -> Atmost:
    computeBheat = (atmost.bheat1.shape[0] != 0)

    z1 = np.zeros((atmost.time.shape[0], newGrid.shape[0]))
    d1 = np.zeros((atmost.time.shape[0], newGrid.shape[0]))
    ne1 = np.zeros((atmost.time.shape[0], newGrid.shape[0]))
    tg1 = np.zeros((atmost.time.shape[0], newGrid.shape[0]))
    vz1 = np.zeros((atmost.time.shape[0], newGrid.shape[0]))
    bheat1 = np.zeros((atmost.time.shape[0], newGrid.shape[0]))
    nh1 = np.zeros((atmost.time.shape[0], 6, newGrid.shape[0]))

    for i in range(z1.shape[0]):
        z1[i] = np.interp(newGrid, cmass[i], atmost.z1[i])
        d1[i] = np.interp(newGrid, cmass[i], atmost.d1[i])
        ne1[i] = np.interp(newGrid, cmass[i], atmost.ne1[i])
        tg1[i] = np.interp(newGrid, cmass[i], atmost.tg1[i])
        vz1[i] = np.interp(newGrid, cmass[i], atmost.vz1[i])
        if computeBheat:
            bheat1[i] = np.interp(newGrid, cmass[i], atmost.bheat1[i])
        for j in range(nh1.shape[1]):
            nh1[i,j] = np.interp(newGrid, cmass[i], atmost.nh1[i,j])

    vturb = np.interp(newGrid, cmass[0], atmost.vturb)

    return Atmost(atmost.grav, atmost.tau2, vturb, atmost.time, atmost.dt, z1, d1, ne1, tg1, vz1, nh1, bheat1, cgs=atmost.cgs)

def interp_to_const_height_grid(atmost: Atmost, newGrid: np.ndarray, interp_fn=np.interp) -> Atmost:
    computeBheat = (atmost.bheat1.shape[0] != 0)

    z1 = np.zeros((atmost.time.shape[0], newGrid.shape[0]))
    d1 = np.zeros((atmost.time.shape[0], newGrid.shape[0]))
    ne1 = np.zeros((atmost.time.shape[0], newGrid.shape[0]))
    tg1 = np.zeros((atmost.time.shape[0], newGrid.shape[0]))
    vz1 = np.zeros((atmost.time.shape[0], newGrid.shape[0]))
    bheat1 = np.zeros((atmost.time.shape[0], newGrid.shape[0]))
    nh1 = np.zeros((atmost.time.shape[0], 6, newGrid.shape[0]))

    for i in range(z1.shape[0]):
        z1[i] = newGrid
        d1[i] = interp_fn(newGrid, atmost.z1[i], atmost.d1[i])
        ne1[i] = interp_fn(newGrid, atmost.z1[i], atmost.ne1[i])
        tg1[i] = interp_fn(newGrid, atmost.z1[i], atmost.tg1[i])
        vz1[i] = interp_fn(newGrid, atmost.z1[i], atmost.vz1[i])
        if computeBheat:
            bheat1[i] = interp_fn(newGrid, atmost.z1[i], atmost.bheat1[i])
        for j in range(nh1.shape[1]):
            nh1[i,j] = interp_fn(newGrid, atmost.z1[i], atmost.nh1[i,j])

    vturb = interp_fn(newGrid, atmost.z1[0], atmost.vturb)

    return Atmost(atmost.grav, atmost.tau2, vturb, atmost.time, atmost.dt, z1, d1, ne1, tg1, vz1, nh1, bheat1, cgs=atmost.cgs)