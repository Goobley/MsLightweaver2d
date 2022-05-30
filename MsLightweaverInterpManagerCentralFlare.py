from multiprocessing.sharedctypes import Value
import pickle
import numpy as np
import matplotlib.pyplot as plt
from lightweaver.rh_atoms import H_6_atom, C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, Fe_atom, FeI_atom, MgII_atom, N_atom, Na_atom, S_atom, CaII_atom
from lightweaver.atmosphere import Atmosphere, ScaleType
from lightweaver.atomic_table import DefaultAtomicAbundance
from lightweaver.atomic_set import RadiativeSet, SpeciesStateTable
from lightweaver.molecule import MolecularTable
from lightweaver.LwCompiled import LwContext
from lightweaver.utils import InitialSolution, planck, NgOptions, ConvergenceError
import lightweaver.constants as Const
import lightweaver as lw
from typing import List
from copy import deepcopy
from MsLightweaverAtoms import H_6, CaII, H_6_nasa, CaII_nasa
import os
import os.path as path
import time
from radynpy.matsplotlib import OpcFile
from radynpy.utils import hydrogen_absorption
from numba import njit
from pathlib import Path
from scipy.linalg import solve
from scipy.interpolate import interp1d, PchipInterpolator
import warnings
from weno4 import weno4
import pdb
from copy import copy
import zarr

def interp(xs, xp, fp):
    # order = np.argsort(xp)
    # xp = np.ascontiguousarray(xp[order])
    # fp = np.ascontiguousarray(fp[order])

    # return np.interp(xs, xp, fp)
    return weno4(xs, xp, fp)

# https://stackoverflow.com/a/21901260
import subprocess
def mslightweaver_revision():
    p = Path(__file__).parent
    isGitRepo = subprocess.check_output(['git', 'rev-parse', '--is-inside-work-tree'], cwd=p).decode('ascii').strip() == 'true'
    if not isGitRepo:
        raise ValueError('Cannot find git info.')

    gitChanges = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=p).decode('ascii').strip()
    if len(gitChanges) > 0:
        raise ValueError('Uncommitted changes to tracked files, cannot procede:\n%s' % gitChanges)

    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=p).decode('ascii').strip()

def check_write_git_revision(outputDir):
    revision = mslightweaver_revision()
    with open(outputDir + 'GitRevision.txt', 'w') as f:
        f.write(revision)

def FastBackground(*args):
    import lightweaver.LwCompiled
    return lightweaver.LwCompiled.FastBackground(*args, Nthreads=16)

@njit
def time_dep_update_impl(theta, dt, Gamma, GammaPrev, n, nPrev):
    Nlevel = n.shape[0]
    Nspace = n.shape[1]

    GammaPrev = GammaPrev if GammaPrev is not None else np.empty_like(Gamma)
    Gam = np.zeros((Nlevel, Nlevel))
    nk = np.zeros(Nlevel)
    nPrevIter = np.zeros(Nlevel)
    nCurrent = np.zeros(Nlevel)
    atomDelta = 0.0

    for k in range(Nspace):
        nCurrent[:] = n[:, k]
        nPrevIter[:] = nPrev[:, k]
        Gam[...] = -theta * Gamma[:,:, k] * dt
        Gam += np.eye(Nlevel)
        if theta != 1.0:
            nk[:] = (1.0 - theta) * dt * GammaPrev[:,:, k] @ nPrevIter + nPrevIter
        else:
            nk[:] = nPrevIter

        nNew = np.linalg.solve(Gam, nk)
        n[:, k] = nNew
        atomDelta = max(atomDelta, np.nanmax(np.abs(1.0 - nCurrent / nNew)))

    return atomDelta

class MsLightweaverInterpManager:
    def __init__(self, atmost, outputDir,
                 atoms,
                 fixedZGrid,
                 activeAtoms=['H', 'Ca'],
                 maxZ=None,
                 startingCtx=None, conserveCharge=False,
                 prd=False, Nthreads=16):
        # check_write_git_revision(outputDir)
        self.atmost = atmost
        self.outputDir = outputDir
        self.conserveCharge = conserveCharge
        self.abund = DefaultAtomicAbundance
        self.idx = 0
        self.nHTot = atmost.d1 / (self.abund.massPerH * Const.Amu)
        self.prd = prd
        self.updateRhoPrd = False
        self.Nthreads = Nthreads

        self.fixedZGrid = fixedZGrid
        self.Nz = fixedZGrid.shape[0]
        self.maxZ = maxZ

        self.zarrStore = zarr.convenience.open(outputDir + 'MsLw1dCentral.zarr')
        if startingCtx is not None:
            self.ctx = startingCtx
            args = startingCtx.kwargs
            self.atmos = args['atmos']
            self.spect = args['spect']
            self.aSet = self.spect.radSet
            self.eqPops = args['eqPops']
            self.nltePopsStore = self.zarrStore['SimOutput/Populations/NLTE']
            self.ltePopsStore = self.zarrStore['SimOutput/Populations/LTE']
            self.radStore = self.zarrStore['SimOutput/Radiation']
            self.neStore = self.zarrStore['SimOutput/ne']
            self.zGridStore = self.zarrStore['SimOutput/zAxis']
        else:
            temperature = weno4(self.fixedZGrid, atmost.z1[0], atmost.tg1[0])
            vlos = weno4(self.fixedZGrid, atmost.z1[0], atmost.vz1[0])
            vturb = np.ones_like(vlos) * 2e3
            ne1 = weno4(self.fixedZGrid, atmost.z1[0], atmost.ne1[0])
            nHTot = weno4(self.fixedZGrid, atmost.z1[0], self.nHTot[0])
            self.atmos = Atmosphere.make_1d(scale=ScaleType.Geometric, depthScale=np.copy(self.fixedZGrid), temperature=np.copy(temperature),
                                            vlos=np.copy(vlos), vturb=np.copy(vturb), ne=np.copy(ne1), nHTot=nHTot)

            self.atmos.quadrature(5)

            self.aSet = RadiativeSet(atoms)
            self.aSet.set_active(*activeAtoms)

            self.spect = self.aSet.compute_wavelength_grid()

            if self.conserveCharge:
                self.eqPops = self.aSet.iterate_lte_ne_eq_pops(self.atmos)
            else:
                self.eqPops = self.aSet.compute_eq_pops(self.atmos)

            self.ctx = lw.Context(self.atmos, self.spect, self.eqPops,
                                  initSol=InitialSolution.Lte,
                                  conserveCharge=self.conserveCharge,
                                  Nthreads=self.Nthreads,
                                  backgroundProvider=FastBackground, hprd=prd)
            simOut = self.zarrStore.require_group('SimOutput')
            pops = simOut.require_group('Populations')
            self.nltePopsStore = pops.require_group('NLTE')
            self.ltePopsStore = pops.require_group('LTE')
            self.radStore = simOut.require_group('Radiation')
            simParams = self.zarrStore.require_group('SimParams')
            simParams['wavelength'] = self.ctx.spect.wavelength
            simParams['zAxisInitial'] = np.copy(self.fixedZGrid)

            self.radStore['J'] = np.zeros((0, *self.ctx.spect.J.shape))
            self.radStore['I'] = np.zeros((0, *self.ctx.spect.I.shape))
            simOut['zAxis'] = np.zeros((0, self.fixedZGrid.shape[0]))
            self.zGridStore = simOut['zAxis']
            for atom in self.eqPops.atomicPops:
                if atom.pops is not None:
                    self.ltePopsStore[atom.element.name] = np.zeros((0, *atom.nStar.shape))
                    self.nltePopsStore[atom.element.name] = np.zeros((0, *atom.pops.shape))
            simOut['ne'] = np.zeros((0, *self.atmos.ne.shape))
            self.neStore = simOut['ne']

        self.atmos.bHeat = np.ones(self.atmos.Nspace) * 1e-20
        self.atmos.hPops = self.eqPops['H']


    def initial_stat_eq(self, Nscatter=3, NmaxIter=1000, popsTol=1e-3, JTol=3e-3,
                        overwritePops=True):
        # if self.prd:
        #     self.ctx.configure_hprd_coeffs()

        lw.iterate_ctx_se(self.ctx, prd=self.prd, Nscatter=Nscatter, NmaxIter=NmaxIter, 
                          popsTol=popsTol, JTol=JTol)


    def save_timestep(self, forceOverwrite=False):
        if self.idx == 0 and self.radStore['I'].shape[0] > 0 and not forceOverwrite:
            return
        self.radStore['J'].append(np.expand_dims(self.ctx.spect.J, 0))
        self.radStore['I'].append(np.expand_dims(self.ctx.spect.I, 0))
        for atom in self.eqPops.atomicPops:
            if atom.pops is not None:
                self.ltePopsStore[atom.element.name].append(np.expand_dims(atom.nStar, 0))
                self.nltePopsStore[atom.element.name].append(np.expand_dims(atom.pops, 0))
        self.neStore.append(np.expand_dims(self.atmos.ne, 0))
        self.zGridStore.append(np.expand_dims(self.fixedZGrid, 0))


    def load_timestep(self, stepNum, destroyLaterSteps=False):
        self.idx = stepNum
        zGrid = self.zGridStore[self.idx]
        zRadyn = self.atmost.z1[self.idx]
        self.atmos.temperature[:] = interp(zGrid, zRadyn, self.atmost.tg1[self.idx])
        self.atmos.vlos[:] = interp(zGrid, zRadyn, self.atmost.vz1[self.idx])

        self.atmos.nHTot[:] = interp(zGrid, zRadyn, self.nHTot[self.idx])
        self.atmos.bHeat[:] = interp(zGrid, zRadyn, self.atmost.bheat1[self.idx])

        for name, pops in self.nltePopsStore.items():
            self.eqPops.atomicPops[name].pops[:] = pops[self.idx]
            # NOTE(cmo): Remove entries after the one being loaded
            if destroyLaterSteps:
                pops.resize(self.idx+1, pops.shape[1], pops.shape[2])

        for name, pops in self.ltePopsStore.items():
            self.eqPops.atomicPops[name].nStar[:] = pops[self.idx]
            if destroyLaterSteps:
                pops.resize(self.idx+1, pops.shape[1], pops.shape[2])

        neStore = self.neStore
        self.atmos.ne[:] = neStore[self.idx]
        if destroyLaterSteps:
            neStore.resize(self.idx+1, *neStore.shape[1:])

        shape = self.radStore['I'].shape
        self.ctx.spect.I[:] = self.radStore['I'][self.idx]
        if destroyLaterSteps:
            self.radStore['I'].resize(self.idx+1, *shape[1:])

        shape = self.radStore['J'].shape
        self.ctx.spect.J[:] = self.radStore['J'][self.idx]
        if destroyLaterSteps:
            self.radStore['J'].resize(self.idx+1, *shape[1:])

        self.ctx.update_deps()

    def increment_step(self, newZGrid):
        self.idx += 1
        prevZGrid = self.fixedZGrid

        self.fixedZGrid = newZGrid
        zGrid = self.fixedZGrid
        zRadyn = self.atmost.z1[self.idx]
        self.atmos.z[:] = zGrid
        self.atmos.temperature[:] = interp(zGrid, zRadyn, self.atmost.tg1[self.idx])
        self.atmos.vlos[:] = interp(zGrid, zRadyn, self.atmost.vz1[self.idx])
        if not self.conserveCharge:
            self.atmos.ne[:] = interp(zGrid, zRadyn, self.atmost.ne1[self.idx])
        else:
            self.atmos.ne[:] = interp(zGrid, prevZGrid, self.atmos.ne)

        self.atmos.nHTot[:] = interp(zGrid, zRadyn, self.nHTot[self.idx])
        self.atmos.bHeat[:] = interp(zGrid, zRadyn, self.atmost.bheat1[self.idx])

        # self.ctx.spect.I[...] = 0.0
        # self.ctx.spect.J[...] = 0.0

        # self.eqPops.update_lte_atoms_Hmin_pops(self.atmos, self.conserveCharge, updateTotals=True)

        for atom in self.eqPops.atomicPops:
            if atom.pops is not None:
                atom.update_nTotal(self.atmos)
                for i in range(atom.pops.shape[0]):
                    atom.pops[i] = interp(zGrid, prevZGrid, atom.pops[i])
                # NOTE(cmo): We have the new nTotal from nHTot after update_deps()
                atom.pops *= (atom.nTotal / np.sum(atom.pops, axis=0))[None, :]

        self.ctx.update_deps()

        # if self.prd:
        #     self.ctx.update_hprd_coeffs()

    def time_dep_prev_state(self, evalGamma=False):
        if evalGamma:
            self.ctx.formal_sol_gamma_matrices()
        s = {}
        s['pops'] = [np.copy(a.n) for a in self.ctx.activeAtoms]
        s['Gamma'] = [np.copy(a.Gamma) if evalGamma else None for a in self.ctx.activeAtoms]
        return s

    def time_dep_update(self, dt, prevState, theta=0.5):
        atoms = self.ctx.activeAtoms
        Nspace = self.atmos.Nspace

        maxDelta = 0.0
        for i, atom in enumerate(atoms):
            atomDelta = time_dep_update_impl(theta, dt, atom.Gamma, prevState['Gamma'][i],
                                             atom.n, prevState['pops'][i])

            maxDelta = max(maxDelta, atomDelta)
            s = '    %s delta = %6.4e' % (atom.atomicModel.element, atomDelta)
            print(s)

        return maxDelta

    def compute_2d_bc_rays(self, muz, wmu):
        atmos = copy(self.atmos)
        atmos.rays(muz, wmu=2.0*wmu)
        print('------')
        print('ctxRays BC')
        print('------')
        ctxRays = lw.Context(atmos, self.ctx.kwargs['spect'], self.ctx.eqPops, Nthreads=16)
        ctxRays.spect.J[:] = self.ctx.spect.J
        ctxRays.depthData.fill = True
        for i in range(50):
            dJ = ctxRays.formal_sol_gamma_matrices()
            if dJ.dJMax < 1e-3:
                break


        return ctxRays.depthData.I


    def time_dep_step(self, nSubSteps=200, popsTol=1e-3, JTol=3e-3, rhoTol=1e-2, theta=1.0, dt=None, depth=0):
        dt = dt if dt is not None else self.atmost.dt[self.idx+1]
        dNrPops = 0.0
        # self.ctx.spect.J[:] = 0.0
        if self.prd:
            for atom in self.ctx.activeAtoms:
                for t in atom.trans:
                    t.recompute_gII()

        prevState = None
        # prevState = self.time_dep_prev_state(evalGamma=(theta!=1.0))
        prdStartedOnSub = 0
        try:
            for sub in range(nSubSteps):
                if self.updateRhoPrd and sub > 0:
                    dPrd = self.ctx.prd_redistribute(maxIter=10, tol=popsTol)
                    print(dPrd.compact_representation())

                dJ = self.ctx.formal_sol_gamma_matrices()
                print(dJ.compact_representation())
                dJMaxIdx = dJ.dJMaxIdx
                Nz = self.atmost.z1.shape[1]
                laMax = dJMaxIdx
                print(f'{self.ctx.spect.wavelength[laMax]} nm')
                delta, prevState = self.ctx.time_dep_update(dt, prevState)
                # delta = self.time_dep_update(dt, prevState, theta=theta)
                if self.conserveCharge:
                    dNrPops = self.ctx.nr_post_update(timeDependentData={'dt': dt, 'nPrev': prevState})

                popsChange = dNrPops if self.conserveCharge else delta
                print(popsChange.compact_representation())
                dPopsMaxIdx = popsChange.dPopsMaxIdx[0]
                levelMax = dPopsMaxIdx // Nz
                kMax = dPopsMaxIdx - (levelMax * Nz)
                print(f'level={levelMax}, {self.atmos.height[kMax] / 1e6} Mm ({kMax})')


                if sub > 1 and (popsChange.dPopsMax < popsTol and dJ.dJMax < JTol):
                    if self.prd: 
                        if self.updateRhoPrd:
                            if dPrd.dRhoMax < rhoTol and sub - prdStartedOnSub > 1:
                                break
                        else:
                            print('Starting PRD Iterations')
                            self.updateRhoPrd = True
                            prdStartedOnSub = sub
                    else:
                        break
            else:
                raise ValueError('NON-CONVERGED')
        except:
            if depth == 0:
                print('Depth bail-out')
                raise ValueError('Depth didn\'t fix')

            print(f'Retrying finer: depth: {depth+1}')
            self.ctx.time_dep_restore_prev_pops(prevState)
            self.ctx.atmos.ne[:] = weno4(self.ctx.atmos.height, self.zGridStore[-1][:], self.neStore[-1][:])
            self.ctx.update_deps()
            for t in range(10):
                print(f'substep {t}')
                self.time_dep_step(nSubSteps, popsTol, JTol, rhoTol, theta, dt=0.1*dt, depth=depth+1)

def convert_atomic_pops(atom):
    d = {}
    if atom.pops is not None:
        d['n'] = atom.pops
    else:
        d['n'] = atom.pops
    d['nStar'] = atom.nStar
    return d

def distill_pops(eqPops):
    d = {}
    for atom in eqPops.atomicPops:
        d[atom.element.name] = convert_atomic_pops(atom)
    return d
