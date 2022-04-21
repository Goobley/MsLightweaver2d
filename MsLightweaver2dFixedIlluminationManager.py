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
from scipy.signal import wiener
from MsLightweaverInterpManager import MsLightweaverInterpManager
from MsLightweaverInterpQSManager import MsLightweaverInterpQSManager
from MsLightweaverUtil import test_timesteps_in_dir, optional_load_starting_context
from ReadAtmost import read_atmost
from weno4 import weno4
import zarr

def interp(xs, xp, fp):
    # order = np.argsort(xp)
    # xp = np.ascontiguousarray(xp[order])
    # fp = np.ascontiguousarray(fp[order])

    # return np.interp(xs, xp, fp)
    return weno4(xs, xp, fp)

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
    def __init__(self, outputDir, atmost, Nz, xAxis,
                 atoms,
                 activeAtoms=['H', 'Ca'],
                 startingCtx=None, startingCtx1d=None, startingCtxQs=None,
                 conserveCharge=False,
                 saveJ=True,
                 firstColumnFrom1d=False):
        # test_timesteps_in_dir(OutputDir)

        self.atmost = atmost
        self.Nz = Nz
        self.xAxis = xAxis
        self.conserveCharge = conserveCharge
        self.outputDir = outputDir
        self.activeAtoms = activeAtoms
        self.atoms = atoms
        self.saveJ = saveJ
        self.firstColumnFrom1d = firstColumnFrom1d

        # NOTE(cmo): Compute initial z-axis - just respace while keeping same distribution.
        # Actually, try and use the one that will be used in step 1.
        # self.zAxis = np.interp(np.linspace(0, 1, self.Nz),
        #                        np.linspace(0, 1, self.atmost.z1.shape[1]),
        #                        atmost.z1[0])
        self.idx = 1
        self.zAxis = self.next_z_axis()
        self.idx = 0

        zAxis = self.zAxis
        self.nHTotRadyn0 = atmost.d1[0] / (lw.DefaultAtomicAbundance.massPerH * lw.Amu)

        # NOTE(cmo): Initialise 1D boundary condition
        self.ms = MsLightweaverInterpManager(atmost=self.atmost, outputDir=outputDir,
                                atoms=atoms, fixedZGrid=self.zAxis,
                                activeAtoms=activeAtoms, startingCtx=startingCtx1d,
                                conserveCharge=False,
                                # zarrName=None,
                                prd=False)
        self.ms.initial_stat_eq(popsTol=1e-3, Nscatter=10)
        self.ms.save_timestep()
        # NOTE(cmo): QS Bc
        self.msQs = MsLightweaverInterpQSManager(atmost=self.atmost, outputDir=outputDir,
                                                 atoms=atoms, fixedZGrid=self.zAxis,
                                                 activeAtoms=activeAtoms, startingCtx=startingCtxQs, conserveCharge=False,
                                                 prd=False)
        self.msQs.stat_eq(popsTol=1e-3, Nscatter=10)
        self.msQs.save_timestep()

        # NOTE(cmo): Set up 2D atmosphere
        Nz = self.ms.fixedZGrid.shape[0]
        Nx = xAxis.shape[0]
        self.Nz = Nz
        self.Nx = Nx

        Nquad2d = 11
        self.Nquad2d = Nquad2d

        zarrName = None
        zarrName = 'MsLw2d.zarr' if zarrName is None else zarrName
        self.zarrStore = zarr.convenience.open(outputDir + zarrName)
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
            self.zGridStore = self.zarrStore['SimOutput/zAxis']
            self.ctx.update_deps()
            # self.load_timestep(0)
        else:
            temperature = np.zeros((Nz, Nx))
            temperature[...] = self.ms.atmos.temperature[:, None]
            ne = np.zeros((Nz, Nx))
            ne[...] = self.ms.atmos.ne[:, None]
            vz = np.zeros((Nz, Nx))
            vx = np.zeros((Nz, Nx))
            vturb = np.ones((Nz, Nx)) * 2e3
            nHTot = np.zeros((Nz, Nx))
            nHTot[...] = self.ms.atmos.nHTot[:, None]
            self.atmos2d = lw.Atmosphere.make_2d(height=zAxis, x=xAxis, temperature=temperature,
                                            ne=ne, vx=vx, vz=vz, vturb=vturb, nHTot=nHTot,
                                            xLowerBc=FixedXBc('lower'), xUpperBc=FixedXBc('upper'))
            self.eqPops2d = self.ms.aSet.compute_eq_pops(self.atmos2d)
            for atom in activeAtoms:
                self.eqPops2d[atom].reshape(-1, Nz, Nx)[...] = self.ms.eqPops[atom][:, :, None]

            self.atmos2d.hPops = self.eqPops2d['H']
            self.atmos2d.bHeat = np.zeros(Nz * Nx)
            self.atmos2d.quadrature(Nquad2d)
            self.aSet = self.ms.aSet
            # ctx = lw.Context(atmos2d, ms.spect, eqPops2d, Nthreads=70, crswCallback=lw.CrswIterator())
            self.ctx = lw.Context(self.atmos2d, self.ms.spect, self.eqPops2d, Nthreads=72,
                                #   formalSolver='piecewise_linear_2d',
                                  conserveCharge=conserveCharge,
                                  backgroundProvider=FastBackground)

            simParams = self.zarrStore.require_group('SimParams')
            simParams['zAxisInitial'] = np.copy(self.zAxis)
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
            timeData['zAxis'] = zarr.zeros((0, Nz), chunks=(1, Nz))
            self.zGridStore = timeData['zAxis']
            self.timeRadStore = timeData.require_group('Radiation')
            self.timePopsStore = timeData.require_group('Populations')
            # self.timeRadStore['J'] = np.expand_dims(self.ctx.spect.J, 0)
            # self.timeRadStore['I'] = np.expand_dims(self.ctx.spect.I, 0)
            if self.saveJ:
                self.timeRadStore['J'] = zarr.zeros((0, *self.ctx.spect.J.shape), chunks=(1, *self.ctx.spect.J.shape))
            self.timeRadStore['I'] = zarr.zeros((0, *self.ctx.spect.I.shape), chunks=(1, *self.ctx.spect.I.shape))
            self.ltePopsStore = self.timePopsStore.require_group('LTE')
            self.nltePopsStore = self.timePopsStore.require_group('NLTE')
            timeData['ne'] = zarr.zeros((0, *self.atmos2d.ne.shape), chunks=(1, *self.atmos2d.ne.shape))
            self.neStore = timeData['ne']
            for atom in self.eqPops2d.atomicPops:
                if atom.pops is not None:
                    self.ltePopsStore[atom.element.name] = zarr.zeros((0, *atom.nStar.shape), chunks=(1, *atom.nStar.shape))
                    self.nltePopsStore[atom.element.name] = zarr.zeros((0, *atom.pops.shape), chunks=(1, *atom.pops.shape))

        self.idx = self.ms.idx


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
        self.zGridStore.append(np.expand_dims(self.zAxis, 0))

    def load_timestep(self, stepNum):
        self.ms.load_timestep(stepNum)
        self.msQs.load_timestep(stepNum)
        self.idx = self.ms.idx

        for name, pops in self.nltePopsStore.items():
            self.eqPops2d.atomicPops[name].pops[:] = pops[self.idx]
            # NOTE(cmo): Remove entries after the one being loaded
            pops.resize(self.idx+1, pops.shape[1], pops.shape[2])

        for name, pops in self.ltePopsStore.items():
            self.eqPops2d.atomicPops[name].nStar[:] = pops[self.idx]
            pops.resize(self.idx+1, pops.shape[1], pops.shape[2])

        zGridStore = self.zGridStore
        self.atmos2d.z[:] = zGridStore[self.idx]
        zGridStore.resize(self.idx+1, *zGridStore.shape[1:])

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

        if self.firstColumnFrom1d:
            self.copy_first_column_from_1d()
        self.ctx.update_deps()

    def copy_first_column_from_1d(self):
        for name, pops in self.nltePopsStore.items():
            pops1 = self.ms.eqPops[name]
            Nlevel = pops1.shape[0]
            Nz = pops1.shape[1]
            self.eqPops2d.atomicPops[name].pops.reshape(Nlevel,Nz,-1)[:,:,0] = pops1

        for name, pops in self.ltePopsStore.items():
            pops1 = self.ms.eqPops.atomicPops[name].nStar
            Nlevel = pops1.shape[0]
            Nz = pops1.shape[1]
            self.eqPops2d.atomicPops[name].nStar.reshape(Nlevel,Nz,-1)[:,:,0] = pops1

        self.atmos2d.ne.reshape(Nz,-1)[:, 0] = self.ms.atmos.ne[:]
        self.atmos2d.nHTot.reshape(Nz,-1)[:, 0] = self.ms.atmos.nHTot[:]
        self.atmos2d.temperature.reshape(Nz,-1)[:, 0] = self.ms.atmos.temperature[:]
        self.atmos2d.vz.reshape(Nz,-1)[:,0] = self.ms.atmos.vlos[:]


    def increment_step(self):
        self.idx += 1
        # NOTE(cmo): First compute new zGrid for coming step
        prevZAxis = self.zAxis
        self.zAxis = self.next_z_axis()
        zGrid = self.zAxis

        self.ms.increment_step(self.zAxis)
        self.ms.time_dep_step(popsTol=1e-3, JTol=5e-3, nSubSteps=1000, theta=1.0)
        self.ms.save_timestep()

        self.msQs.increment_step(self.zAxis)
        self.msQs.stat_eq(popsTol=1e-3, JTol=5e-3)
        self.msQs.save_timestep()

        Nx = self.atmos2d.Nx

        zRadyn = self.atmost.z1[0]
        self.atmos2d.z[:] = self.zAxis
        temperature = interp(zGrid, zRadyn, self.atmost.tg1[0])
        temp2d = self.atmos2d.temperature.reshape(self.Nz, Nx)
        temp2d[...] = temperature[:, None]

        ne2d = self.atmos2d.ne.reshape(self.Nz, Nx)
        if not self.conserveCharge:
            ne = interp(zGrid, zRadyn, self.atmost.ne1[0])
            ne2d[...] = ne[:, None]
        else:
            for x in range(Nx):
                ne2d[:, x] = interp(zGrid, prevZAxis, ne2d[:, x])

        nHTot = interp(zGrid, zRadyn, self.nHTotRadyn0)
        nHTot2d = self.atmos2d.nHTot.reshape(self.Nz, Nx)
        nHTot2d[...] = nHTot[:, None]

        # self.ctx.spect.I[...] = 0.0
        # self.ctx.spect.J[...] = 0.0

        # if self.conserveCharge:

        # self.eqPops2d.update_lte_atoms_Hmin_pops(self.atmos2d, self.conserveCharge, updateTotals=True)
        for atom in self.eqPops2d.atomicPops:
            atom.update_nTotal(self.atmos2d)
            if atom.pops is not None:
                pops2d = atom.pops.reshape(atom.pops.shape[0], self.Nz, Nx)
                for i in range(pops2d.shape[0]):
                    for x in range(pops2d.shape[2]):
                        pops2d[i, :, x] = interp(zGrid, prevZAxis, pops2d[i, :, x])
                # NOTE(cmo): We have the new nTotal from nHTot after update_deps()
                atom.pops *= (atom.nTotal / np.sum(atom.pops, axis=0))[None, :]

        # NOTE(cmo): If set, load 1D sim into first column
        if self.firstColumnFrom1d:
            self.copy_first_column_from_1d()

        self.ctx.update_deps()


    def initial_stat_eq(self, Nscatter=10, NmaxIter=1000, popsTol=1e-3):
        bcIntensity = self.ms.compute_2d_bc_rays(self.atmos2d.muz[:self.Nquad2d], self.atmos2d.wmu[:self.Nquad2d])
        self.atmos2d.xLowerBc.set_bc(bcIntensity)
        self.atmos2d.xUpperBc.set_bc(bcIntensity)

        lw.iterate_ctx_se(self.ctx, Nscatter=Nscatter, NmaxIter=NmaxIter, popsTol=popsTol)

        self.ms.atmos.bHeat[:] = weno4(self.zAxis, self.ms.atmost.z1[0], self.ms.atmost.bheat1[0])


    def time_dep_step(self, Nsubsteps, popsTol):

        bcIntensity = self.ms.compute_2d_bc_rays(self.atmos2d.muz[:self.Nquad2d], self.atmos2d.wmu[:self.Nquad2d])
        self.atmos2d.xLowerBc.set_bc(bcIntensity)

        qsBcIntensity = self.msQs.compute_2d_bc_rays(self.atmos2d.muz[:self.Nquad2d], self.atmos2d.wmu[:self.Nquad2d])
        self.atmos2d.xUpperBc.set_bc(qsBcIntensity)

        print('-'*40)
        print('1D BC Done')
        print('-'*40)
        # for backgroundIter in range(2):
        #     self.ctx.formal_sol_gamma_matrices()
        prevState = None
        dt = self.ms.atmost.dt[self.idx+1]
        Nz = self.atmos2d.Nz
        Nx = self.atmos2d.Nx
        for iter2d in range(Nsubsteps):
            dJ = self.ctx.formal_sol_gamma_matrices()
            print(dJ.compact_representation())
            # if self.firstColumnFrom1d:
            #     for atom in self.ctx.activeAtoms:
            #         # 0 here gives identity for the time-dependent transition matrix
            #         atom.Gamma.reshape(-1, Nz, Nx)[:,:,0] = 0.0
            if self.firstColumnFrom1d:
                nDagger = []
                neDagger = np.copy(self.atmos2d.ne)

                for atom in self.aSet.activeAtoms:
                    nDagger.append(np.copy(self.eqPops2d[atom.element]))

            dPops, prevState = self.ctx.time_dep_update(dt, prevState, chunkSize=-1)
            if not self.conserveCharge:
                print(dPops.compact_representation())
                dPops = dPops.dPopsMax

            if self.conserveCharge:
                dPops = self.ctx.nr_post_update(timeDependentData={'dt': dt, 'nPrev': prevState}, chunkSize=-1)
                print(dPops.compact_representation())
                dPops = dPops.dPopsMax
                # NOTE(cmo): This is implicitly handled by the "Ng region" now,
                # so dPops will be the change over the total iterative
                # procedure.
                # dPops = max(dPops, dNrPops)

            if self.firstColumnFrom1d:
                self.copy_first_column_from_1d()

                dPops = 0.0
                for i, atom in enumerate(self.aSet.activeAtoms):
                    n = self.eqPops2d[atom.element]
                    nDag = nDagger[i]

                    relChange = np.nanmax(np.abs(n - nDag) / n)
                    dPops = max(dPops, relChange)

                relChange = np.nanmax(np.abs(self.atmos2d.ne - neDagger) / self.atmos2d.ne)
                dPops = max(dPops, relChange)
                print('    dPops after resetting first column to 1D values: %6.4e' % dPops)


            if dPops < popsTol and iter2d > 3:
                break
        else:
            raise ValueError('2D iteration failed to converge')

    def next_z_axis(self):
        idxO = 0
        idxN = self.idx
        DistTol = 1
        PointTotal = self.Nz
        SmoothingSize = 15
        HalfSmoothingSize = SmoothingSize // 2

        # Merge grids
        uniqueCombined = list(np.unique(np.sort(np.concatenate((self.atmost.z1[idxO],
                                                                self.atmost.z1[idxN])))))
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
        z1O = np.copy(self.atmost.z1[idxO][::-1])
        startIdx = np.searchsorted(z1O, uniqueCombined[0])
        endIdx = np.searchsorted(z1O, uniqueCombined[-1])
        for v in z1O[:startIdx]:
            uniqueCombined.append(v)
        for v in z1O[endIdx:]:
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
