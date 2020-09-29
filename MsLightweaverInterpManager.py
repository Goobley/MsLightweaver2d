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
# from HydroWeno.Simulation import Grid
# from HydroWeno.Advector import Advector
# from HydroWeno.BCs import zero_grad_bc
# from HydroWeno.Weno import reconstruct_weno_nm_z
import warnings
from traceback import print_stack
from weno4 import weno4
from RadynAdvection import an_sol, an_rad_sol, an_gml_sol
import pdb
from copy import copy

def weno4_pos(xs, xp, fp, **kwargs):
    return np.exp(weno4_safe(xs, xp, np.log(fp), **kwargs))


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

def nr_advect(atmost, i0, eqPops, activeAtomNames, abundances):
    d1 = atmost.d1[i0+1]
    for a in activeAtomNames:
        pop = np.zeros_like(eqPops[a])
        for i in range(pop.shape[0]):
            pop[i, :] = an_sol(atmost, i0, eqPops[a][i], tol=1e-8, maxIter=1000)
        nTotal = d1 / (abundances.massPerH * lw.Amu) * abundances[a]
        popCorrectionFactor = nTotal / pop.sum(axis=0)
        print('Max Correction %s: %.2e' % (a, np.abs(1-popCorrectionFactor).max()))
        pop *= popCorrectionFactor
        eqPops[a][...] = pop

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
                 fixedZGrid, atoms,
                 activeAtoms=['H', 'Ca'],
                 startingCtx=None, conserveCharge=False,
                 prd=False):
        # check_write_git_revision(outputDir)
        self.atmost = atmost
        self.outputDir = outputDir
        self.conserveCharge = conserveCharge
        self.abund = DefaultAtomicAbundance
        self.idx = 0
        self.nHTot = atmost.d1 / (self.abund.massPerH * Const.Amu)
        self.prd = prd
        self.fixedZGrid = fixedZGrid

        if startingCtx is not None:
            self.ctx = startingCtx
            args = startingCtx.arguments
            self.atmos = args['atmos']
            self.spect = args['spect']
            self.aSet = self.spect.radSet
            self.eqPops = args['eqPops']
        else:
            temperature = weno4(self.fixedZGrid, atmost.z1[0], atmost.tg1[0])
            vlos = weno4(self.fixedZGrid, atmost.z1[0], atmost.vz1[0])
            vturb = np.ones_like(vlos) * 2e3
            ne1 = weno4(self.fixedZGrid, atmost.z1[0], atmost.ne1[0])
            nHTot = weno4(self.fixedZGrid, atmost.z1[0], self.nHTot[0])
            self.atmos = Atmosphere.make_1d(scale=ScaleType.Geometric, depthScale=np.copy(self.fixedZGrid), temperature=np.copy(temperature),
                                            vlos=np.copy(vlos), vturb=np.copy(vturb), ne=np.copy(ne1), nHTot=nHTot)

            # self.atmos.convert_scales()
            self.atmos.quadrature(5)

            self.aSet = RadiativeSet(atoms)
            self.aSet.set_active(*activeAtoms)
            # NOTE(cmo): Radyn seems to compute the collisional rates once per
            # timestep(?) and we seem to get a much better agreement for Ca
            # with the CH rates when H is set to LTE for the initial timestep.
            # Might be a bug in my implementation though.

            self.spect = self.aSet.compute_wavelength_grid()

            self.mols = MolecularTable()
            if self.conserveCharge:
                self.eqPops = self.aSet.iterate_lte_ne_eq_pops(self.atmos, self.mols)
            else:
                self.eqPops = self.aSet.compute_eq_pops(self.atmos, self.mols)

            self.ctx = lw.Context(self.atmos, self.spect, self.eqPops, initSol=InitialSolution.Lte, conserveCharge=self.conserveCharge, Nthreads=16)

        self.atmos.bHeat = np.ones(self.atmos.Nspace) * 1e-20
        self.atmos.hPops = self.eqPops['H']
        np.save(self.outputDir + 'Wavelength.npy', self.ctx.spect.wavelength)


    def initial_stat_eq(self, Nscatter=3, NmaxIter=1000, popTol=1e-3, JTol=3e-3):
        if self.prd:
            self.ctx.configure_hprd_coeffs()

        for i in range(NmaxIter):
            dJ = self.ctx.formal_sol_gamma_matrices()
            if i < Nscatter:
                continue

            delta = self.ctx.stat_equil()
            if self.prd:
                self.ctx.prd_redistribute()

            if self.ctx.crswDone and dJ < JTol and delta < popTol:
                print('Stat eq converged in %d iterations' % (i+1))
                break
        else:
            raise ConvergenceError('Stat Eq did not converge.')

    def save_timestep(self):
        i = self.idx
        with open(self.outputDir + 'Step_%.6d.pickle' % i, 'wb') as pkl:
            eqPops = distill_pops(self.eqPops)
            Iwave = self.ctx.spect.I
            pickle.dump({'eqPops': eqPops, 'Iwave': Iwave, 'ne': self.atmos.ne}, pkl)

    # def load_timestep(self, stepNum):
    #     with open(self.outputDir + 'Step_%.6d.pickle' % stepNum, 'rb') as pkl:
    #         step = pickle.load(pkl)

    #     self.idx = stepNum
    #     self.atmos.temperature[:] = self.atmost.tg1[self.idx]
    #     self.atmos.vlos[:] = self.atmost.vz1[self.idx]
    #     if not self.conserveCharge:
    #         self.atmos.ne[:] = self.atmost.ne1[self.idx]

    #     if self.advectPops or self.rescalePops:
    #         self.atmos.nHTot[:] = self.nHTot[self.idx]
    #     self.atmos.bHeat[:] = self.atmost.bheat1[self.idx]

    #     self.atmos.height[:] = self.atmost.z1[self.idx]

    #     for name, pops in step['eqPops'].items():
    #         if pops['n'] is not None:
    #             self.eqPops.atomicPops[name].pops[:] = pops['n']
    #         self.eqPops.atomicPops[name].nStar[:] = pops['nStar']
    #     self.atmos.ne[:] = step['ne']
    #     self.ctx.spect.I[:] = step['Iwave']
    #     self.ctx.update_deps()

    def increment_step(self):
        self.idx += 1
        self.atmos.temperature[:] = weno4(self.fixedZGrid, self.atmost.z1[self.idx], self.atmost.tg1[self.idx])
        self.atmos.vlos[:] = weno4(self.fixedZGrid, self.atmost.z1[self.idx], self.atmost.vz1[self.idx])
        if not self.conserveCharge:
            self.atmos.ne[:] = weno4(self.fixedZGrid, self.atmost.z1[self.idx], self.atmost.ne1[self.idx])

        self.atmos.nHTot[:] = weno4(self.fixedZGrid, self.atmost.z1[self.idx], self.nHTot[self.idx])
        self.atmos.bHeat[:] = weno4(self.fixedZGrid, self.atmost.z1[self.idx], self.atmost.bheat1[self.idx])

        # self.atmos.height[:] = self.atmost.z1[self.idx]
        self.ctx.update_deps()
        if self.prd:
            self.ctx.configure_hprd_coeffs()
        # self.opac_background()

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
        ctxRays.depthData.fill = True
        dJ = 1.0
        while dJ > 1.0e-3:
            dJ = ctxRays.formal_sol_gamma_matrices()

        return ctxRays.depthData.I


    def time_dep_step(self, nSubSteps=200, popsTol=1e-3, JTol=3e-3, theta=1.0, dt=None):
        dt = dt if dt is not None else self.atmost.dt[self.idx+1]
        dNrPops = 0.0
        underTol = False
        # self.ctx.spect.J[:] = 0.0
        if self.prd:
            for atom in self.ctx.activeAtoms:
                for t in atom.trans:
                    try:
                        t.rhoPrd.fill(1.0)
                        t.gII[0,0,0] = -1.0
                    except:
                        pass

            self.ctx.configure_hprd_coeffs()
            self.ctx.formal_sol_gamma_matrices()
            self.ctx.prd_redistribute(200)

        prevState = self.time_dep_prev_state(evalGamma=(theta!=1.0))
        for sub in range(nSubSteps):
            # self.JPrev[:] = self.ctx.spect.J
            if self.prd and sub > 1:
                if delta > 5e-1:
                    # self.ctx.prd_redistribute(maxIter=2, tol=5e-1)
                    pass
                else:
                    self.ctx.prd_redistribute(maxIter=5, tol=min(1e-1, 10*delta))

            dJ = self.ctx.formal_sol_gamma_matrices()
            # if sub > 2:
            # delta, prevState = self.ctx.time_dep_update(dt, prevState)
            delta = self.time_dep_update(dt, prevState, theta=theta)
            # for ia, atom in enumerate(self.ctx.activeAtoms):
            #     nDag = np.copy(atom.n)
            #     print(atom.atomicModel.name)
            #     gml = np.zeros_like(atom.n)
            #     for i in range(atom.n.shape[0]):
            #         for k in range(atom.n.shape[1]):
            #             gml[i, k] = atom.Gamma[i, :, k] @ atom.n[:, k]

            #     for i in range(atom.n.shape[0]):
            #         atom.n[i, :] = an_gml_sol(self.atmost, self.idx, prevState['pops'][ia][i], atom.n[i], gml[i], tol=1e-8)

            #     advChange = np.abs(nDag - atom.n) / nDag
            #     print('advChange: %.3e' % advChange.max())


            #     # delta = an_rad_sol(self.atmost, self.idx, prevState['pops'][ia], atom.n, atom.nTotal, atom.Gamma)

            # pdb.set_trace()
            # delta = advChange.max()
            if delta > 5e-1:
                # underTol = False
                continue
            if not underTol:
                underTol = True
                if sub > 0 and self.conserveCharge:
                    # self.eqPops.update_lte_atoms_Hmin_pops(self.atmos, True, True)
                    self.ctx.update_deps()
                    # if self.prd:
                        # self.ctx.configure_hprd_coeffs()

            if self.conserveCharge:
                dNrPops = self.ctx.nr_post_update(timeDependentData={'dt': dt, 'nPrev': prevState['pops']})



            if sub > 1 and ((delta < popsTol and dJ < JTol and dNrPops < popsTol)
                            or (delta < 0.1*popsTol and dNrPops < 0.1*popsTol)
                            or (dJ < 1e-6)):
                break
        else:
            self.ctx.depthData.fill = True
            self.ctx.formal_sol_gamma_matrices()
            self.ctx.depthData.fill = False

            sourceData = {
                'chi': np.copy(self.ctx.depthData.chi),
                'eta': np.copy(self.ctx.depthData.eta),
                'chiBg': np.copy(self.ctx.background.chi),
                'etaBg': np.copy(self.ctx.background.eta),
                'scaBg': np.copy(self.ctx.background.sca),
                'J': np.copy(self.ctx.spect.J)
                         }
            with open(self.outputDir + 'Fails.txt', 'a') as f:
                f.write('%d, %.4e %.4e\n' % (self.idx, delta, dJ))

            with open(self.outputDir + 'NonConvergenceData_%.6d.pickle' % (self.idx), 'wb') as pkl:
                pickle.dump(sourceData, pkl)

            print('NON-CONVERGED')
        # self.ctx.time_dep_conserve_charge(prevState)


    # def time_dep_step(self, nSubSteps=200, popsTol=1e-3, JTol=3e-3, theta=0.5):
    #     dt = self.atmost.dt[self.idx+1]
    #     dNrPops = 0.0
    #     underTol = False

    #     prevState = self.time_dep_prev_state()
    #     for sub in range(nSubSteps):
    #         dJ = self.ctx.formal_sol_gamma_matrices()
    #         delta = self.time_dep_update(0.5 * dt, prevState, theta=theta)
    #         if sub > 1 and ((delta < popsTol and dJ < JTol and dNrPops < popsTol)
    #                         or (delta < 0.1*popsTol and dNrPops < 0.1*popsTol)
    #                         or (dJ < 1e-6)):
    #             break

    #     print('-'*80)
    #     print('ADVECTING')
    #     print('-'*80)
    #     nr_advect(self.atmost, self.idx, self.eqPops, [a.name for a in self.aSet.activeAtoms], self.at)

    #     prevState = self.time_dep_prev_state()
    #     for sub in range(nSubSteps):
    #         dJ = self.ctx.formal_sol_gamma_matrices()
    #         delta = self.time_dep_update(0.5 * dt, prevState, theta=theta)
    #         if sub > 1 and ((delta < popsTol and dJ < JTol and dNrPops < popsTol)
    #                         or (delta < 0.1*popsTol and dNrPops < 0.1*popsTol)
    #                         or (dJ < 1e-6)):
    #             break

    def cont_fn_data(self, step):
        self.load_timestep(step)
        self.ctx.depthData.fill = True
        dJ = 1.0
        while dJ > 1e-3:
            dJ = self.ctx.formal_sol_gamma_matrices()
        self.ctx.depthData.fill = False
        J = np.copy(self.ctx.spect.J)

        sourceData = {'chi': np.copy(self.ctx.depthData.chi),
                      'eta': np.copy(self.ctx.depthData.eta),
                      'chiBg': np.copy(self.ctx.background.chi),
                      'etaBg': np.copy(self.ctx.background.eta),
                      'scaBg': np.copy(self.ctx.background.sca),
                      'J': J
                      }
        return sourceData

    def rf_k(self, step, dt, pertSize, k, Jstart=None):
        self.load_timestep(step)
        print(pertSize)

        self.ctx.clear_ng()
        if Jstart is not None:
            self.ctx.spect.J[:] = Jstart
        else:
            self.ctx.spect.J[:] = 0.0

        self.atmos.temperature[k] += 0.5 * pertSize
        self.ctx.update_deps()

        if Jstart is None:
            dJ = 1.0
            while dJ > 1e-3:
                dJ = self.ctx.formal_sol_gamma_matrices()
        self.time_dep_step(popsTol=1e-3, JTol=5e-3, dt=dt, theta=1.0)
        plus = np.copy(self.ctx.spect.I[:, -1])

        self.load_timestep(step)
        self.ctx.clear_ng()
        if Jstart is not None:
            self.ctx.spect.J[:] = Jstart
        else:
            self.ctx.spect.J[:] = 0.0

        self.atmos.temperature[k] -= 0.5 * pertSize
        self.ctx.update_deps()

        if Jstart is None:
            dJ = 1.0
            while dJ > 1e-3:
                dJ = self.ctx.formal_sol_gamma_matrices()
        self.time_dep_step(popsTol=1e-3, JTol=5e-3, dt=dt, theta=1.0)
        minus = np.copy(self.ctx.spect.I[:, -1])

        return plus, minus

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
