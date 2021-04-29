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
import zarr
from copy import copy

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

class MsLw2d:
    def __init__(self, outputDir, atmost, zAxis, xAxis,
                 slabData,
                 atoms,
                 activeAtoms=['H', 'Ca'],
                 startingCtx=None, startingCtx1d=None,
                 conserveCharge=False,
                 saveJ=True):
        # test_timesteps_in_dir(OutputDir)

        self.atmost = atmost
        self.zAxis = zAxis
        self.xAxis = xAxis
        self.conserveCharge = conserveCharge
        self.outputDir = outputDir
        self.activeAtoms = activeAtoms
        self.atoms = atoms
        self.saveJ = saveJ

        # NOTE(cmo): Initialise 1D boundary condition
        self.ms = MsLightweaverInterpManager(atmost=self.atmost, outputDir=outputDir,
                                atoms=atoms, fixedZGrid=self.zAxis,
                                activeAtoms=activeAtoms, startingCtx=startingCtx1d,
                                conserveCharge=False,
                                prd=False)
        self.ms.initial_stat_eq(popTol=1e-3, Nscatter=10)
        self.ms.save_timestep()
        self.slabData = slabData
        rightBcAtmos = lw.Atmosphere.make_1d(scale=lw.ScaleType.Geometric, depthScale=self.zAxis,
                                             temperature=slabData['temperature'],
                                             vlos=np.zeros_like(slabData['temperature']),
                                             vturb=np.ones_like(slabData['temperature']) * 2e3,
                                             ne=slabData['ne'],
                                             nHTot=slabData['nHTot'])
        rightBcAtmos.quadrature(5)
        self.rAtmos = rightBcAtmos
        raSet = lw.RadiativeSet(atoms)
        raSet.set_active(*activeAtoms)
        rSpect = raSet.compute_wavelength_grid()
        rEqPops = raSet.compute_eq_pops(rightBcAtmos)
        rightBcAtmos.hPops = rEqPops['H']
        rightBcAtmos.bHeat = np.zeros_like(rightBcAtmos.temperature)
        rCtx = lw.Context(rightBcAtmos, rSpect, rEqPops,
                          initSol=lw.InitialSolution.Lte,
                          conserveCharge=False,
                          Nthreads=16)
        self.rCtx = rCtx
        for i in range(1000):
            dJ = rCtx.formal_sol_gamma_matrices()
            if i < 10:
                continue

            delta = rCtx.stat_equil()
            if dJ < 3e-3 and delta < 1e-3:
                print('Right boundary converged in %d iterations' % i)
                break
        else:
            raise lw.ConvergenceError('Stat eq did not converge.')


        # NOTE(cmo): Set up 2D atmosphere
        Nz = self.ms.fixedZGrid.shape[0]
        Nx = xAxis.shape[0]
        self.Nz = Nz
        self.Nx = Nx

        Nquad2d = 6
        self.Nquad2d = Nquad2d

        self.zarrStore = zarr.convenience.open(outputDir + 'MsLw2d.zarr')
        if startingCtx is not None:
            self.ctx = startingCtx
            args = startingCtx.kwargs
            self.atmos2d = args['atmos']
            self.spect = args['spect']
            self.aSet = self.spect.radSet
            self.eqPops2d = args['eqPops']
            self.nltePopsStore = self.zarrStore['SimOutput/Populations/NLTE']
            self.ltePopsStore = self.zarrStore['SimOutput/Populations/LTE']
            self.timeRadStore = self.zarrStore['SimOutput/Radiation']
            self.neStore = self.zarrStore['SimOutput/ne']
            self.ctx.update_deps()
            # self.load_timestep(0)
        else:
            temperature = np.zeros((Nz, Nx))
            temperature[...] = slabData['temperature'][:, None]
            ne = np.zeros((Nz, Nx))
            ne[...] = slabData['ne'][:, None]
            vz = np.zeros((Nz, Nx))
            vx = np.zeros((Nz, Nx))
            vturb = np.ones((Nz, Nx)) * 2e3
            nHTot = np.zeros((Nz, Nx))
            nHTot[...] = slabData['nHTot'][:, None]
            self.atmos2d = lw.Atmosphere.make_2d(height=zAxis, x=xAxis, temperature=temperature,
                                            ne=ne, vx=vx, vz=vz, vturb=vturb, nHTot=nHTot,
                                            xLowerBc=FixedXBc('lower'), xUpperBc=FixedXBc('upper'))
            self.eqPops2d = self.ms.aSet.compute_eq_pops(self.atmos2d)
            for atom in activeAtoms:
                self.eqPops2d[atom].reshape(-1, Nz, Nx)[...] = self.ms.eqPops[atom][:, :, None]

            self.atmos2d.hPops = self.eqPops2d['H']
            self.atmos2d.bHeat = np.zeros(Nz * Nx)
            self.atmos2d.quadrature(Nquad2d)
            # ctx = lw.Context(atmos2d, ms.spect, eqPops2d, Nthreads=70, crswCallback=lw.CrswIterator())
            self.ctx = lw.Context(self.atmos2d, self.ms.spect, self.eqPops2d, Nthreads=72,
                                  formalSolver='piecewise_linear_2d', conserveCharge=conserveCharge,
                                  backgroundProvider=FastBackground)

            simParams = self.zarrStore.require_group('SimParams')
            simParams['zAxis'] = self.zAxis
            simParams['xAxis'] = self.xAxis
            simParams['wavelength'] = self.ctx.spect.wavelength
            ics = self.zarrStore.require_group('InitialConditions')
            atmos = self.atmos2d.dimensioned_view()
            ics['temperature'] = atmos.temperature
            ics['vz'] = atmos.vz
            ics['vx'] = atmos.vx
            ics['vturb'] = atmos.vturb
            ics['ne'] = atmos.ne
            ics['nHTot'] = atmos.nHTot
            # ics['xLowerBc'] = atmos.xLowerBc.I
            # ics['xUpperBc'] = atmos.xUpperBc.I

            timeData = self.zarrStore.require_group('SimOutput')
            self.timeRadStore = timeData.require_group('Radiation')
            self.timePopsStore = timeData.require_group('Populations')
            # self.timeRadStore['J'] = np.expand_dims(self.ctx.spect.J, 0)
            # self.timeRadStore['I'] = np.expand_dims(self.ctx.spect.I, 0)
            if self.saveJ:
                self.timeRadStore['J'] = np.zeros((0, *self.ctx.spect.J.shape))
            self.timeRadStore['I'] = np.zeros((0, *self.ctx.spect.I.shape))
            self.ltePopsStore = self.timePopsStore.require_group('LTE')
            self.nltePopsStore = self.timePopsStore.require_group('NLTE')
            timeData['ne'] = np.zeros((0, *self.atmos2d.ne.shape))
            self.neStore = timeData['ne']
            for atom in self.eqPops2d.atomicPops:
                if atom.pops is not None:
                    self.ltePopsStore[atom.element.name] = np.zeros((0, *atom.nStar.shape))
                    self.nltePopsStore[atom.element.name] = np.zeros((0, *atom.pops.shape))

        self.idx = self.ms.idx


    def compute_right_bc_rays(self, muz, wmu):
        atmos = copy(self.rAtmos)
        atmos.rays(muz, wmu=2.0*wmu)
        print('------')
        print('ctxRays BC')
        print('------')
        ctxRays = lw.Context(atmos, self.rCtx.kwargs['spect'], self.rCtx.eqPops, Nthreads=16)
        ctxRays.spect.J[:] = self.rCtx.spect.J
        ctxRays.depthData.fill = True
        for i in range(50):
            dJ = ctxRays.formal_sol_gamma_matrices()
            if dJ < 1e-3:
                break


        return ctxRays.depthData.I


    def save_timestep_data(self):
        if self.idx == 0 and self.timeRadStore['I'].shape[0] > 0:
            return

        if self.saveJ:
            self.timeRadStore['J'].append(np.expand_dims(self.ctx.spect.J, 0))
        self.timeRadStore['I'].append(np.expand_dims(self.ctx.spect.I, 0))
        for atom in self.eqPops2d.atomicPops:
            if atom.pops is not None:
                self.ltePopsStore[atom.element.name].append(np.expand_dims(atom.nStar, 0))
                self.nltePopsStore[atom.element.name].append(np.expand_dims(atom.pops, 0))
        self.neStore.append(np.expand_dims(self.atmos2d.ne, 0))

    def load_timestep(self, stepNum):
        self.ms.load_timestep(stepNum)
        self.idx = self.ms.idx

        for name, pops in self.nltePopsStore.items():
            self.eqPops2d.atomicPops[name].pops[:] = pops[self.idx]
            # NOTE(cmo): Remove entries after the one being loaded
            pops.resize(self.idx+1, pops.shape[1], pops.shape[2])

        for name, pops in self.ltePopsStore.items():
            self.eqPops2d.atomicPops[name].nStar[:] = pops[self.idx]
            pops.resize(self.idx+1, pops.shape[1], pops.shape[2])

        neStore = self.neStore
        self.atmos2d.ne[:] = neStore[self.idx]
        neStore.resize(self.idx+1, *neStore.shape[1:])

        shape = self.timeRadStore['I'].shape
        self.ctx.spect.I[:] = self.timeRadStore['I'][self.idx]
        self.timeRadStore['I'].resize(self.idx+1, *shape[1:])

        if self.saveJ:
            shape = self.timeRadStore['J'].shape
            self.ctx.spect.J[:] = self.timeRadStore['J'][self.idx]
            self.timeRadStore['J'].resize(self.idx+1, *shape[1:])

        self.ctx.update_deps()

    def increment_step(self):
        self.idx += 1
        if self.conserveCharge:
            self.ctx.update_deps()
        self.ms.increment_step()
        self.ms.time_dep_step(popsTol=1e-3, JTol=5e-3, nSubSteps=1000, theta=1.0)
        self.ms.save_timestep()


    def initial_stat_eq(self, Nscatter=10, NmaxIter=1000, popTol=1e-3):
        bcIntensity = self.ms.compute_2d_bc_rays(self.atmos2d.muz[:self.Nquad2d], self.atmos2d.wmu[:self.Nquad2d])
        rightBcIntensity = self.compute_right_bc_rays(self.atmos2d.muz[:self.Nquad2d], self.atmos2d.wmu[:self.Nquad2d])
        self.atmos2d.xLowerBc.set_bc(bcIntensity)
        self.atmos2d.xUpperBc.set_bc(rightBcIntensity)
        for i in range(Nscatter):
            self.ctx.formal_sol_gamma_matrices()
        for i in range(2000):
            self.ctx.formal_sol_gamma_matrices()
            dPops = self.ctx.stat_equil(chunkSize=-1)
            if dPops < popTol and i > 3 and self.ctx.crswDone:
                break

        self.ms.atmos.bHeat[:] = weno4(self.zAxis, self.ms.atmost.z1[0], self.ms.atmost.bheat1[0])


    def time_dep_step(self, Nsubsteps, popsTol):
        bcIntensity = self.ms.compute_2d_bc_rays(self.atmos2d.muz[:self.Nquad2d], self.atmos2d.wmu[:self.Nquad2d])
        self.atmos2d.xLowerBc.set_bc(bcIntensity)
        print('-------')
        print('1D BC Done')
        print('-------')
        for backgroundIter in range(2):
            self.ctx.formal_sol_gamma_matrices()
        prevState = None
        dt = self.ms.atmost.dt[self.idx+1]
        for iter2d in range(Nsubsteps):
            self.ctx.formal_sol_gamma_matrices()
            dPops, prevState = self.ctx.time_dep_update(dt, prevState, chunkSize=-1)
            if self.conserveCharge:
                dPops = self.ctx.nr_post_update(timeDependentData={'dt': dt, 'nPrev': prevState}, chunkSize=-1)
                # NOTE(cmo): This is implicitly handled by the "Ng region" now,
                # so dPops will be the change over the total iterative
                # procedure.
                # dPops = max(dPops, dNrPops)
            if dPops < popsTol and iter2d > 5:
                break
        else:
            raise ValueError('2D iteration failed to converge')
