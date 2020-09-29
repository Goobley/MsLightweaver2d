import pickle
import numpy as np
import matplotlib.pyplot as plt
from lightweaver.rh_atoms import H_6_atom, C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, Fe_atom, FeI_atom, MgII_atom, N_atom, Na_atom, S_atom, CaII_atom, He_9_atom
import lightweaver as lw
from MsLightweaverAtoms import H_6, CaII, H_6_nasa, CaII_nasa
from pathlib import Path
import os
import os.path as path
import time
from MsLightweaverInterpManager import MsLightweaverInterpManager
from MsLightweaverUtil import test_timesteps_in_dir, optional_load_starting_context
from ReadAtmost import read_atmost
from weno4 import weno4

OutputDir = '2DTimesteps/'
Path(OutputDir).mkdir(parents=True, exist_ok=True)
Path(OutputDir + '/Rfs').mkdir(parents=True, exist_ok=True)
Path(OutputDir + '/ContFn').mkdir(parents=True, exist_ok=True)
NasaAtoms = [H_6_nasa(), CaII_nasa(), He_9_atom(), C_atom(), O_atom(), Si_atom(), Fe_atom(),
             MgII_atom(), N_atom(), Na_atom(), S_atom()]
FchromaAtoms = [H_6(), CaII(), He_9_atom(), C_atom(), O_atom(), Si_atom(), Fe_atom(),
                MgII_atom(), N_atom(), Na_atom(), S_atom()]
AtomSet = FchromaAtoms
ConserveCharge = False
PopulationTransportMode = None
Prd = False

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

    def set_bc(self, data):
        self.I = I1d.reshape(I1d.shape[0], -1, I1d.shape[-1])

    def compute_bc(self, atmos, spect):
        # if spect.wavelength.shape[0] != self.I.shape[0]:
        #     result = np.ones((spect.wavelength.shape[0], spect.I.shape[1], atmos.Nz))
        # else:
        if self.I is None:
            raise ValueError('I has not been set (x%sBc)' % self.mode)
        result = np.copy(self.I)
        return result

test_timesteps_in_dir(OutputDir)

atmost = read_atmost('atmost.dat')
atmost.to_SI()
if atmost.bheat1.shape[0] == 0:
    atmost.bheat1 = np.load('BheatInterp.npy')

with open('ZGrid2d.pkl', 'rb') as pkl:
    zAxis = pickle.load(pkl)['zAxis']

startingCtx = optional_load_starting_context(OutputDir)

start = time.time()
# NOTE(cmo): Initialise 1D boundary condition
ms = MsLightweaverInterpManager(atmost=atmost, outputDir=OutputDir,
                          atoms=AtomSet, fixedZGrid=zAxis,
                          activeAtoms=['H', 'Ca'], startingCtx=startingCtx,
                          conserveCharge=ConserveCharge,
                          prd=Prd)
ms.initial_stat_eq(popTol=1e-3, Nscatter=10)
ms.save_timestep()

# NOTE(cmo): Set up 2D atmosphere
xAxis = np.linspace(0, 200e3, 20)
Nz = ms.fixedZGrid.shape[0]
Nx = xAxis.shape[0]
temperature = np.zeros((Nz, Nx))
temperature[...] = ms.atmos.temperature[:, None]
ne = np.zeros((Nz, Nx))
ne[...] = ms.atmos.ne[:, None]
vz = np.zeros((Nz, Nx))
vx = np.zeros((Nz, Nx))
vturb = np.ones((Nz, Nx)) * 2e3
nHTot = np.zeros((Nz, Nx))
nHTot[...] = ms.atmos.nHTot[:, None]
atmos2d = lw.Atmosphere.make_2d(height=ms.fixedZGrid, x=xAxis, temperature=temperature,
                                vx=vx, vz=vz, vturb=vturb, nHTot=nHTot,
                                xLowerBc=FixedXBc('lower'), xUpperBc=FixedXBc('upper'))
eqPops2d = ms.aSet.compute_eq_pops(atmos2d)
atmos2d.hPops = eqPops2d['H']
atmos2d.bHeat = np.zeros(Nz * Nx)
Nquad2d = 6
atmos2d.quadrature(Nquad2d)
ctx = lw.Context(atmos2d, ms.spect, eqPops2d, Nthreads=64)
# NOTE(cmo): Initial stat-eq in 2D atmosphere
bcIntensity = ms.compute_2d_bc_rays(atmos2d.muz[:Nquad2d], atmos2d.wmu[:Nquad2d])
atmos2d.xLowerBc.set_bc(bcIntensity)
atmos2d.xUpperBc.set_bc(bcIntensity)
for i in range(5):
    ctx.formal_sol_gamma_matrices()
for i in range(1000):
    ctx.formal_sol_gamma_matrices()
    dPops = ctx.stat_equil()
    if dPops < 1e-3 and i > 5:
        break

maxSteps = ms.atmost.time.shape[0] - 1
ms.atmos.bHeat[:] = weno4(ms.fixedZGrid, ms.atmost.z1[0], ms.atmost.bheat1[0])
firstStep = 0
for i in range(firstStep, maxSteps):
    stepStart = time.time()
    if i != 0:
        ms.increment_step()
    ms.time_dep_step(popsTol=1e-3, JTol=5e-3, nSubSteps=1000, theta=1.0)
    ms.save_timestep()

    bcIntensity = ms.compute_2d_bc_rays(atmos2d.muz[:Nquad2d], atmos2d.wmu[:Nquad2d])
    atmos2d.xLowerBc.set_bc(bcIntensity)
    atmos2d.xUpperBc.set_bc(bcIntensity)
    print('-------')
    print('1D BC Done')
    print('-------')
    for i in range(2):
        ctx.formal_sol_gamma_matrices()
    prevState = None
    for i in range(1000):
        ctx.time_dep_update()
        dPops, prevState = ctx.time_dep_update(ms.atmost.dt[ms.idx+1], prevState)
        if dPops < 1e-3 and i > 5:
            break

    stepEnd = time.time()
    print('-------')
    print('Timestep %d done (%f s)' % ((i+1), ms.atmost.time[i+1]))
    print('Time taken for step %.2e s' % (stepEnd - stepStart))
    print('-------')