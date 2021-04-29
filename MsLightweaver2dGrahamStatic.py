import pickle
import numpy as np
import matplotlib.pyplot as plt
from lightweaver.rh_atoms import H_6_atom, C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, Fe_atom, FeI_atom, MgII_atom, N_atom, Na_atom, S_atom, CaII_atom, He_9_atom
import lightweaver as lw
from MsLightweaverAtoms import H_6, CaII, H_6_nasa, CaII_nasa, H_6_nobb
from pathlib import Path
import os
import os.path as path
import time
from ReadAtmost import read_atmost
from RadynEmistab import EmisTable
from radynpy.cdf import LazyRadynData
import astropy.units as u
from copy import copy
from weno4 import weno4

class CoronalIrraditation(lw.BoundaryCondition):
    def __init__(self):
        # NOTE(cmo): This data needs to be in (mu, toObs) order, i.e. mu[0]
        # down, mu[0] up, mu[1] down...
        # self.I = I1d.reshape(I1d.shape[0], -1, I1d.shape[-1])
        self.I = None

    def set_bc(self, I1d):
        if I1d.ndim == 3:
            self.I = I1d
        else:
            self.I = np.expand_dims(I1d, axis=2)

    def compute_bc(self, atmos, spect):
        # if spect.wavelength.shape[0] != self.I.shape[0]:
        #     result = np.ones((spect.wavelength.shape[0], spect.I.shape[1], atmos.Nz))
        # else:
        if self.I is None:
            raise ValueError('I has not been set (CoronalIrradtion)')
        result = np.copy(self.I)
        return result

radynData = LazyRadynData('QSHTSL_1F11_d5_ec20_t10s.cdf')
Idx = 96 # 8 s
ConvergenceTol = 1e-3

mergedZAxis = np.copy(np.unique(np.sort(np.concatenate((radynData.z1[0] / 1e2, radynData.z1[Idx] / 1e2))))[::-1])
MinSpacing = 5e2
pointsTooClose = np.where(np.abs(np.diff(mergedZAxis)) < MinSpacing)[0]
while pointsTooClose.shape[0] > 0:
    newAxis = np.zeros(mergedZAxis.shape[0] - pointsTooClose.shape[0])
    pointsTooClose += 1
    i = 0
    for j in range(mergedZAxis.shape[0]):
        if j in pointsTooClose:
            continue
        newAxis[i] = mergedZAxis[j]
        i += 1
    mergedZAxis = newAxis
    pointsTooClose = np.where(np.abs(np.diff(mergedZAxis)) < MinSpacing)[0]

NasaAtoms = [H_6_nasa(), CaII_nasa(), He_9_atom(), C_atom(), O_atom(), Si_atom(), Fe_atom(),
             MgII_atom(), N_atom(), Na_atom(), S_atom()]
FchromaAtoms = [H_6(), CaII(), He_9_atom(), C_atom(), O_atom(), Si_atom(), Fe_atom(),
                MgII_atom(), N_atom(), Na_atom(), S_atom()]
atoms = NasaAtoms

nHTot = ((radynData.d1 << u.g / u.cm**3) << (u.kg / u.m**3)).value / (lw.DefaultAtomicAbundance.massPerH * lw.Amu)
activeAtmos = lw.Atmosphere.make_1d(scale=lw.ScaleType.Geometric,
                                    depthScale=mergedZAxis,
                                    temperature=weno4(mergedZAxis, radynData.z1[Idx] / 1e2, radynData.tg1[Idx]),
                                    vlos=weno4(mergedZAxis, radynData.z1[Idx] / 1e2, radynData.vz1[Idx]) << u.cm / u.s,
                                    vturb=np.ones_like(mergedZAxis) * 2e3,
                                    ne=weno4(mergedZAxis, radynData.z1[Idx] / 1e2, radynData.ne1[Idx]) << u.cm**-3,
                                    nHTot=weno4(mergedZAxis, radynData.z1[Idx] / 1e2, nHTot[Idx]),
                                    # upperBc=CoronalIrraditation()
                                    )
activeAtmos.quadrature(5)
preAtmos = lw.Atmosphere.make_1d(scale=lw.ScaleType.Geometric,
                                 depthScale=mergedZAxis,
                                 temperature=weno4(mergedZAxis, radynData.z1[0] / 1e2, radynData.tg1[0]),
                                 vlos=weno4(mergedZAxis, radynData.z1[0] / 1e2, radynData.vz1[0]) << u.cm / u.s,
                                 vturb=np.ones_like(mergedZAxis) * 2e3,
                                 ne=weno4(mergedZAxis, radynData.z1[0] / 1e2, radynData.ne1[0]) << u.cm**-3,
                                 nHTot=weno4(mergedZAxis, radynData.z1[0] / 1e2, nHTot[0]),
                                #  upperBc=CoronalIrraditation()
                                 )
preAtmos.quadrature(5)

aSet = lw.RadiativeSet(atoms)
aSet.set_active('H', 'Ca', 'He')
spect = aSet.compute_wavelength_grid()
eqPops = aSet.compute_eq_pops(activeAtmos)
eqPopsPre = aSet.compute_eq_pops(preAtmos)
ctx = lw.Context(activeAtmos, spect, eqPops, Nthreads=32)
ctxPre = lw.Context(preAtmos, spect, eqPopsPre, Nthreads=32)

activeAtmos.bHeat = np.ones_like(mergedZAxis) * 1e-20
activeAtmos.hPops = eqPops['H']
preAtmos.bHeat = np.ones_like(mergedZAxis) * 1e-20
preAtmos.hPops = eqPopsPre['H']

eqPopsH = ((np.copy(radynData.n1[Idx, :, :6, 0].T) << u.cm**-3) << u.m**-3).value
for i in range(6):
    eqPops['H'][i] = weno4(mergedZAxis, radynData.z1[Idx] / 1e2, eqPopsH[i])
eqPopsCa = ((np.copy(radynData.n1[Idx, :, :6, 1].T) << u.cm**-3) << u.m**-3).value
for i in range(6):
    eqPops['Ca'][i] = weno4(mergedZAxis, radynData.z1[Idx] / 1e2, eqPopsCa[i])
eqPopsHe = ((np.copy(radynData.n1[Idx, :, :, 2].T) << u.cm**-3) << u.m**-3).value
for i in range(9):
    eqPops['He'][i] = weno4(mergedZAxis, radynData.z1[Idx] / 1e2, eqPopsHe[i])

eqPopsH = ((np.copy(radynData.n1[0, :, :6, 0].T) << u.cm**-3) << u.m**-3).value
for i in range(6):
    eqPopsPre['H'][i] = weno4(mergedZAxis, radynData.z1[0] / 1e2, eqPopsH[i])
eqPopsCa = ((np.copy(radynData.n1[0, :, :6, 1].T) << u.cm**-3) << u.m**-3).value
for i in range(6):
    eqPopsPre['Ca'][i] = weno4(mergedZAxis, radynData.z1[0] / 1e2, eqPopsCa[i])
eqPopsHe = ((np.copy(radynData.n1[0, :, :, 2].T) << u.cm**-3) << u.m**-3).value
for i in range(9):
    eqPopsPre['He'][i] = weno4(mergedZAxis, radynData.z1[0] / 1e2, eqPopsHe[i])
irradiationTable = EmisTable('emistab.dat')

# activeAtmos.zUpperBc.set_bc(irradiationTable.compute_downgoing_radiation(spect.wavelength, activeAtmos))
# preAtmos.zUpperBc.set_bc(irradiationTable.compute_downgoing_radiation(spect.wavelength, preAtmos))

for i in range(1000):
    dJ = ctx.formal_sol_gamma_matrices()
    if dJ < ConvergenceTol:
        break

for i in range(1000):
    dJ = ctxPre.formal_sol_gamma_matrices()
    if dJ < ConvergenceTol:
        break

np.save('Wavelength.npy', spect.wavelength)
np.save('I1d.npy', ctx.spect.I)
np.save('I1dPre.npy', ctxPre.spect.I)

def compute_2d_bc_rays(ctx, muz, wmu):
    atmos = copy(ctx.kwargs['atmos'])
    # downRad = np.copy(atmos.zUpperBc.I)
    # downRad = (downRad * ctx.kwargs['atmos'].muz[None, :])[:, 0, 0]
    atmos.rays(muz, wmu=2.0*wmu)
    # atmos.structure.zUpperBc = CoronalIrraditation()
    # atmos.zUpperBc.set_bc(downRad[:, None] / atmos.muz[None, :])
    spect = ctx.kwargs['spect']
    eqPops = ctx.eqPops

    print('------')
    print('ctxRays BC')
    print('------')
    ctxRays = lw.Context(atmos, spect, eqPops, Nthreads=32)
    ctxRays.spect.J[:] = ctx.spect.J
    ctxRays.depthData.fill = True
    for i in range(500):
        dJ = ctxRays.formal_sol_gamma_matrices()
        if dJ < ConvergenceTol:
            break

    return ctxRays.depthData.I

class FixedXBc(lw.BoundaryCondition):
    def __init__(self, mode):
        modes = ['lower', 'upper']
        if not any(mode == m for m in modes):
            raise ValueError('Invalid mode')

        self.mode = mode
        # NOTE(cmo): This data needs to be in (mu, toObs) order, i.e. mu[0]
        # down, mu[0] up, mu[1] down...
        # self.I = I1d.reshape(I1d.shape[0], -1, I1d.shape[-1])
        self.I = None

    def set_bc(self, I1d):
        self.I = I1d.reshape(I1d.shape[0], -1, I1d.shape[-1])

    def compute_bc(self, atmos, spect):
        # if spect.wavelength.shape[0] != self.I.shape[0]:
        #     result = np.ones((spect.wavelength.shape[0], spect.I.shape[1], atmos.Nz))
        # else:
        if self.I is None:
            raise ValueError('I has not been set (x%sBc)' % self.mode)
        result = np.copy(self.I)
        return result


def FastBackground(*args):
    import lightweaver.LwCompiled
    return lightweaver.LwCompiled.FastBackground(*args, Nthreads=72)

Nz = mergedZAxis.shape[0]
Nx = 20
Nquad2d = 6
zAxis = mergedZAxis
xAxis = np.linspace(0, 1e6, Nx)
temperature = np.zeros((Nz, Nx))
temperature[...] = weno4(zAxis, radynData.z1[0] / 1e2, radynData.tg1[0])[:, None]
ne = np.zeros((Nz, Nx))
ne[...] = weno4(zAxis, radynData.z1[0] / 1e2, ((radynData.ne1[0] << u.cm**-3) << u.m**-3).value)[:, None]
vz = np.zeros((Nz, Nx))
vx = np.zeros((Nz, Nx))
vturb = np.ones((Nz, Nx)) * 2e3
nHTot2d = np.zeros((Nz, Nx))
nHTot2d[...] = weno4(zAxis, radynData.z1[0] / 1e2, nHTot[0])[:, None]
atmos2d = lw.Atmosphere.make_2d(height=zAxis, x=xAxis, temperature=temperature,
                                ne=ne, vx=vx, vz=vz, vturb=vturb, nHTot=nHTot2d,
                                # zUpperBc=CoronalIrraditation(),
                                xLowerBc=FixedXBc('lower'), xUpperBc=FixedXBc('upper'))
aSet2d = lw.RadiativeSet(atoms)
aSet2d.set_active('H', 'He', 'Ca')
spect2d = aSet.compute_wavelength_grid()
eqPops2d = aSet2d.compute_eq_pops(atmos2d)
for atom in ['H', 'He', 'Ca']:
    eqPops2d[atom].reshape(-1, Nz, Nx)[...] = eqPopsPre[atom][:, :, None]

atmos2d.hPops = eqPops2d['H']
atmos2d.bHeat = np.zeros(Nz * Nx)
atmos2d.quadrature(Nquad2d)
ctx2d = lw.Context(atmos2d, spect2d, eqPops2d, Nthreads=72,
                   formalSolver='piecewise_linear_2d',
                   backgroundProvider=FastBackground
                   )
ctx2d.spect.J.reshape(-1, Nz, Nx)[...] = ctxPre.spect.J.reshape(-1, Nz)[:, :, None]

atmos2d.xLowerBc.set_bc(compute_2d_bc_rays(ctx, atmos2d.muz[:Nquad2d], atmos2d.wmu[:Nquad2d]))
atmos2d.xUpperBc.set_bc(compute_2d_bc_rays(ctxPre, atmos2d.muz[:Nquad2d], atmos2d.wmu[:Nquad2d]))
# preDowngoingRad = irradiationTable.compute_downgoing_radiation(spect.wavelength, preAtmos)
# preDowngoingJ = (preDowngoingRad * preAtmos.muz[None, :])[:, 0]
# downgoingRad = preDowngoingJ[:, None] / np.abs(atmos2d.muz[None, :])
# downgoingRad2d = np.zeros((*downgoingRad.shape[:2], Nx))
# downgoingRad2d[...] = downgoingRad2d[:, :, 0, None]
# atmos2d.zUpperBc.set_bc(downgoingRad2d)

for i in range(1000):
    dJ = ctx2d.formal_sol_gamma_matrices()
    if i < 10:
        continue
    dPops = ctx2d.stat_equil()
    if dJ < ConvergenceTol and dPops < ConvergenceTol:
        break
ctx2d.depthData.fill = True
ctx2d.formal_sol_gamma_matrices()
