import numpy as np
from numba import njit
from lightweaver import ConvergenceError
from scipy.linalg import solve_banded
import pdb

@njit(cache=True)
def anj_pro(z, vz, dt, i0, an0, an1):
    i1 = i0 + 1
    # z0 = atmost.z1[i0]
    # z1 = atmost.z1[i1]
    # # d0 = atmost.d1[i0]
    # # d1 = atmost.d1[i1]
    # vz0 = atmost.vz1[i0]
    # vz1 = atmost.vz1[i1]
    # dtp = atmost.dt[i0]
    # dtn = atmost.dt[i1]
    # z0 = z[i0]
    # z1 = z[i1]
    # vz0 = vz[i0]
    # vz1 = vz[i1]
    # dtn = dt[i1]
    z0 = z[0]
    z1 = z[1]
    vz0 = vz[0]
    vz1 = vz[1]
    dtn = dt
    theta = 0.55

    dzz = (theta * vz1 + (1-theta)*vz0) * dtn - (z1 - z0)
    # sw = np.where(dzz < 0, -0.5, 0.5)
    sw = np.empty_like(dzz)
    for k in range(z0.shape[0]):
        if dzz[k] < 0:
            sw[k] = -0.5
        else:
            sw[k] = 0.5

    anj = np.empty_like(an0)
    anj[:-1] = theta * ((0.5 + sw[:-1])*an1[1:] + (0.5 - sw[:-1])*an1[:-1]) \
                + (1-theta)*((0.5+sw[:-1])*an0[1:] + (0.5-sw[:-1])*an0[:-1])
    anj[-1] = theta * an1[-1] + (1-theta) * an0[-1]

    da1 = np.empty_like(an0)
    da0 = np.empty_like(an0)
    da1[:-1] = an1[1:] - an1[:-1]
    da1[-1] = 0
    da0[:-1] = an0[1:] - an0[:-1]
    da0[-1] = 0


    # sw2 = np.where((da1 * np.roll(da1,1)) > 0, 1, 0)
    # sw20 = np.where((da0 * np.roll(da0,1)) > 0, 1, 0)
    sw2 = np.zeros_like(da1)
    sw20 = np.zeros_like(da0)
    for k in range(1, z0.shape[0]):
        if da1[k] * da1[k-1] > 0:
            sw2[k] = 1
        if da0[k] * da0[k-1] > 0:
            sw20[k] = 1


    ad1 = np.empty_like(an0)
    ad0 = np.empty_like(an0)
    ad1[1:-1] = (da1[1:-1] + da1[:-2]) * sw2[1:-1] + (1-sw2[1:-1])
    ad0[1:-1] = (da0[1:-1] + da0[:-2]) *sw20[1:-1] + (1-sw20[1:-1])

    da1l = np.empty_like(an0)
    da0l = np.empty_like(an1)
    da1l[1:-1] = da1[1:-1] * da1[:-2] * sw2[1:-1] / ad1[1:-1]
    da0l[1:-1] = da0[1:-1] * da0[:-2] * sw20[1:-1] / ad0[1:-1]
    da1l[-1] = 0.0
    da0l[-1] = 0.0

    anj[1:-1] -= (0.5+sw[1:-1]) * (theta*da1l[2:] + (1-theta)*da0l[2:])
    anj[1:-1] += (0.5-sw[1:-1]) * (theta*da1l[1:-1] + (1-theta)*da0l[1:-1])

    return anj


@njit(cache=True)
def flux_calc_pro(z, d, vz, dt, i0):
    i1 = i0 + 1
    # z0 = atmost.z1[i0]
    # z1 = atmost.z1[i1]
    # d0 = atmost.d1[i0]
    # d1 = atmost.d1[i1]
    # vz0 = atmost.vz1[i0]
    # vz1 = atmost.vz1[i1]
    # dtp = atmost.dt[i0]
    # dtn = atmost.dt[i1]

    z0 = z[i0]
    z1 = z[i1]
    d0 = d[i0]
    d1 = d[i1]
    vz0 = vz[i0]
    vz1 = vz[i1]
    dtn = dt[i1]
    theta = 0.55

    dzz = (theta * vz1 + (1-theta)*vz0) * dtn - (z1 - z0)
    sw = np.where(dzz < 0, -0.5, 0.5)

    dnj = np.zeros_like(d0)
    dnj[:-1] = theta * ((0.5 + sw[:-1])*d1[1:] + (0.5 - sw[:-1])*d1[:-1]) \
                + (1-theta)*((0.5+sw[:-1])*d0[1:] + (0.5-sw[:-1])*d0[:-1])
    dnj[-1] = theta * d1[-1] + (1-theta) * d0[-1]

    dd1 = np.zeros_like(d0)
    dd0 = np.zeros_like(d0)
    dd1[:-1] = d1[1:] - d1[:-1]
    dd0[:-1] = d0[1:] - d0[:-1]

    sw2 = np.where((dd1 * np.roll(dd1,1)) > 0, 1, 0)
    sw20 = np.where((dd0 * np.roll(dd0,1)) > 0, 1, 0)

    ad1 = np.zeros_like(d0)
    ad0 = np.zeros_like(d0)
    ad1[1:-1] = (dd1[1:-1] + dd1[:-2]) * sw2[1:-1] + (1-sw2[1:-1])
    ad0[1:-1] = (dd0[1:-1] + dd0[:-2]) *sw20[1:-1] + (1-sw20[1:-1])

    dd1l = np.zeros_like(d0)
    dd0l = np.zeros_like(d0)
    dd1l[1:-1] = dd1[1:-1] * dd1[:-2] * sw2[1:-1] / ad1[1:-1]
    dd0l[1:-1] = dd0[1:-1] * dd0[:-2] * sw20[1:-1] / ad0[1:-1]
    dd1l[-1] = 0.0
    dd0l[-1] = 0.0

    dnj[1:-1] -= (0.5+sw[1:-1]) * (theta*dd1l[2:] + (1-theta)*dd0l[2:])
    dnj[1:-1] += (0.5-sw[1:-1]) * (theta*dd1l[1:-1] + (1-theta)*dd0l[1:-1])

    fmj2 = dnj * dzz

    return fmj2

def fd_fn(qNew, fn, E=None):
    StencilSize = 5
    UpperBands = StencilSize // 2
    if E is not None:
        initial = E
    else:
        initial = fn(qNew)
    jacobian = np.zeros((StencilSize, initial.shape[0]))
    alpha = 1e-4
    beta = 0.0
    qOrig = qNew.copy()
    pert = qNew * alpha
    for start in range(StencilSize):
        for i in range(start, qNew.shape[0], StencilSize):
            qNew[i] += pert[i]

        objPertP = fn(qNew)
        diff = objPertP - initial

        for i in range(start, qNew.shape[0], StencilSize):
            for s in range(-UpperBands, UpperBands+1):
                if (i+s >= diff.shape[0]) or (i+s < 0):
                    continue
                jacobian[UpperBands + s, i] = diff[i+s] / pert[i]

        qNew[:] = qOrig

    jacobian[2,0] = 1.0

    return jacobian

@njit(cache=True)
def banded_mat_vec(a, b):
    Nbands = a.shape[0]
    UpperBands = a.shape[0] // 2
    assert a.shape[1] == b.shape[0]
    result = np.zeros_like(b)

    for i in range(a.shape[1]):
        for s in range(-UpperBands, UpperBands+1):
            if (i+s >= b.shape[0]) or (i+s < 0):
                continue

        result[i] += a[UpperBands + s, i] * b[i+s]

    return result


def an_sol(atmost, i0, n0, maxIter=100, tol=1e-5):
    i1 = i0 + 1
    z0 = atmost.z1[i0]
    z1 = atmost.z1[i1]
    d0 = atmost.d1[i0]
    d1 = atmost.d1[i1]
    vz0 = atmost.vz1[i0]
    vz1 = atmost.vz1[i1]
    dtn = atmost.dt[i1]
    theta = 0.55

    fmj = flux_calc_pro(atmost.z1, atmost.d1, atmost.vz1, atmost.dt, i0)
    an0 = n0 / d0

    dn0 = np.zeros_like(d0)
    dn0[1:] = n0[1:] * (z0[:-1] - z0[1:])

    def objective(n1):
        an1 = n1 / d1

        anj = anj_pro(atmost.z1[i0:i0+2], atmost.vz1[i0:i0+2], atmost.dt[i1], i0, an0, an1)

        e = np.zeros_like(n1)
        e[1:] = (n1[1:] * (z1[:-1] - z1[1:]) - n0[1:] * (z0[:-1] - z0[1:])) / dn0[1:]
        e[1:] += (anj[:-1] * fmj[:-1] - anj[1:] * fmj[1:]) / dn0[1:]

        return e

    n1Guess = np.copy(n0)

    update = 1

    # NOTE(cmo): Armijo line search parameters
    armijoC = 0.5
    tau = 0.5

    # while update > 1e-5:
    # NOTE(cmo): Primary NR loop
    for i in range(maxIter):
        E = objective(n1Guess)
        W = fd_fn(n1Guess, fn=objective, E=E)
        dq = solve_banded((2, 2), W, -E)
        update = np.abs(dq / n1Guess).max()
        if update < tol:
            break

        # NOTE(cmo): Always do full NR on first iteration, backtrack afterwards
        if i >= 1:
            linearDescent = armijoC * banded_mat_vec(W, dq)
            alpha = 1.0
            for j in range(20):
                possibleObjective = objective(n1Guess + alpha * dq)
                if np.abs(possibleObjective).sum() <= np.abs(E + alpha * linearDescent).sum():
                    dq = alpha * dq
                    break

                alpha *= tau

        # pdb.set_trace()

        n1Guess += dq

    else:
        raise ConvergenceError('Too many iterations')

    # NOTE(cmo): Boundary condition as per RADYN
    n1Guess[0] = n1Guess[1]
    return n1Guess