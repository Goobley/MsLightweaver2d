import pickle
import numpy as np
import matplotlib.pyplot as plt
from lightweaver.rh_atoms import H_6_atom, C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, Fe_atom, FeI_atom, MgII_atom, N_atom, Na_atom, S_atom, CaII_atom
from lightweaver.atmosphere import Atmosphere, ScaleType
from lightweaver.atomic_set import RadiativeSet, SpeciesStateTable
from lightweaver.atomic_table import get_global_atomic_table
from lightweaver.molecule import MolecularTable
from lightweaver.LwCompiled import LwContext
from lightweaver.utils import InitialSolution, planck, NgOptions, ConvergenceError
import lightweaver.constants as Const
import lightweaver as lw
from typing import List
from copy import deepcopy
from MsLightweaverAtoms import H_6, CaII, He_9, H_6_nasa, CaII_nasa
import os
import os.path as path
import time
from notify_run import Notify
from radynpy.matsplotlib import OpcFile
from radynpy.utils import hydrogen_absorption
from numba import njit
from pathlib import Path
from scipy.linalg import solve
from scipy.interpolate import interp1d, PchipInterpolator
from HydroWeno.Simulation import Grid
from HydroWeno.Advector import Advector
from HydroWeno.BCs import zero_grad_bc
from HydroWeno.Weno import reconstruct_weno_nm_z
import warnings
from traceback import print_stack
from weno4 import weno4

def weno4_safe(xs, xp, fp, **kwargs):
    xsort = np.argsort(xp)
    xps = xp[xsort]
    fps = fp[xsort]
    return weno4(xs, xps, fps, **kwargs)

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

@njit
def find_subarray(a, b):
    '''
    Returns the index of the start of b in a, raises ValueError if b is not a
    subarray of a.
    '''

    for i in range(a.shape[0]):
        result = True
        for j in range(b.shape[0]):
            result = result and (a[i+j] == b[j])
            if not result:
                break
        else:
            return i

    raise ValueError

@njit
def construct_common_grid(z0, z1, GridTol=10):
    ClosestPointTol = max(min(np.min(np.abs(np.diff(z0))), np.min(np.abs(np.diff(z1)))), GridTol)
    zComplete = np.concatenate((z0, z1))
    zComplete.sort()
    z = np.zeros_like(zComplete)
    z[0] = zComplete[0]
    mark = 0
    point = 0
    Nspace = 1
    while point < zComplete.shape[0]:
        if zComplete[point] - z[mark] >= ClosestPointTol:
            z[Nspace] = zComplete[point]
            Nspace += 1
            mark += 1

        point += 1

    return z[:Nspace]


def centres_to_interfaces(z):
    return np.concatenate((
        [z[0] - 0.5 * (z[1] - z[0])],
        0.5 * (z[1:] + z[:-1]),
        [z[-1] + 0.5 * (z[-1] - z[-2])]))

def cfl(grid, data):
    vel = np.abs(data[0])
    dt = 0.8 * np.min(grid.dx / vel)
    return dt

def simple_advection_bcs():
    lower_bc = zero_grad_bc('Lower')
    upper_bc = zero_grad_bc('Upper')
    def apply_bcs(grid, V):
        lower_bc(grid, V)
        upper_bc(grid, V)

    return apply_bcs

def advect(atmost, tIdx, eqPops, activeAtomNames, atomicTable, refinementLevel=0):
    NumSubSteps = 5000
    Theta = 0.55
    Refinement = [5, 0.5, 0.05]
    GridTol = Refinement[refinementLevel]
    z0 = atmost.z1[tIdx]
    z1 = atmost.z1[tIdx+1]
    d0 = atmost.d1[tIdx]
    d1 = atmost.d1[tIdx+1]
    numPopRows = 0
    popShapes = []
    for a in activeAtomNames:
        numPopRows += eqPops[a].shape[0]
        popShapes.append(eqPops[a].shape[0])

    dtRadyn = atmost.dt[tIdx+1]
    # zCc = construct_common_grid(z0, z1, GridTol=GridTol)
    zCc = construct_common_grid(z0, z0, GridTol=GridTol)
    zInterfaces = centres_to_interfaces(zCc)
    grid = Grid(zInterfaces, numGhost=4)
    data = np.zeros((1 + numPopRows, grid.griMax))
    def interp(xNew, x, y, **kwargs):
        # return interp1d(x, y, **kwargs, kind=3)(xNew)
        # with warnings.catch_warnings(record=True) as w:
        result = np.exp(interp1d(x, np.log(y), **kwargs, kind=3)(xNew))
            # if len(w) > 0:
            #     print(y)
            #     print_stack()
        return result


    start = 1
    for i, a in enumerate(activeAtomNames):
        # data[start:start+popShapes[i], :] = 10**(interp1d(z0, np.log10(eqPops[a]), fill_value='extrapolate')(grid.cc))
        for j in range(popShapes[i]):
            # data[start+j, :] = interp(grid.cc, z0, eqPops[a][j, :], fill_value='extrapolate')
            # data[start+j, :] = np.exp(PchipInterpolator(z0[::-1], np.log(eqPops[a][j, ::-1]), extrapolate=True)(grid.cc))
            data[start+j, :] = weno4_safe(grid.cc, z0, eqPops[a][j], extrapolate=True)
        # data[start:start+popShapes[i]] /= data[start:start+popShapes[i]].sum(axis=0)
        start += popShapes[i]

    ad = Advector(grid, data, simple_advection_bcs(), reconstruct_weno_nm_z)

    def vel(timestepAdvance=0.0):
        # vz = interp1d(atmost.z1[tIdx], atmost.vz1[tIdx], fill_value='extrapolate')(zCc)
        vz = weno4_safe(zCc, atmost.z1[tIdx], atmost.vz1[tIdx], extrapolate=True)
        if timestepAdvance > 0.0:
            # vz1 = interp1d(atmost.z1[tIdx+1], atmost.vz1[tIdx+1], fill_value='extrapolate')(zCc)
            vz1 = weno4_safe(zCc, atmost.z1[tIdx+1], atmost.vz1[tIdx+1], extrapolate=True)
            cosFac = timestepAdvance
            vz *= (1.0 - cosFac)
            vz += vz1 * cosFac
        return vz
    # print('pre', data[1])

    data[0, grid.griBeg:grid.griEnd] = vel()
    dtMax = cfl(grid, data)
    if dtRadyn > dtMax:
        subTime = 0.0
        for i in range(NumSubSteps):
            dt = dtMax
            if subTime + dt > dtRadyn:
                dt = dtRadyn - subTime

            def update_vel(dt):
                advFrac = (subTime + dt) / dtRadyn
                ad.data[0, grid.griBeg:grid.griEnd] = vel(advFrac)

            ad.step(dt, update_vel=None)

            subTime += dt
            if subTime >= dtRadyn:
                break
            dtMax = cfl(grid, data)
        else:
            raise Exception('Convergence')
    else:
        ad.step(dtRadyn)
    data = ad.data
    if np.any(data[1:] < 0):
        if refinementLevel < len(Refinement) - 1:
            print('Advection failed: trying refinement level %d' % (refinementLevel+1))
            return advect(atmost, tIdx, eqPops, activeAtomNames, atomicTable, refinementLevel=refinementLevel+1)
        else:
            print('Failed at %s' % (repr(np.where(data[1:] < 0))))
            raise ValueError('Advection Failed')
    # print('post', data[1])

    densityAdv = d1 / d0
    start = 1
    for i, a in enumerate(activeAtomNames):
        p = eqPops.atomicPops[a].pops
        # p[:] = 10**(interp1d(grid.cc, np.log10(data[start:start+popShapes[i]]))(z1))
        for j in range(popShapes[i]):
            # p[j, :] = interp(z1, grid.cc, data[start+j])
            # p[j, :] = np.exp(PchipInterpolator(grid.cc, np.log(data[start+j]))(z1))
            p[j, :] = weno4_safe(z1, grid.cc, data[start+j])
        start += popShapes[i]
        nTotal = d1 / (atomicTable.weightPerH * lw.Amu) * atomicTable[a].abundance
        p *= nTotal / p.sum(axis=0)
        # for k in range(p.shape[1]):
        #     maxPop = np.argmax(p[:, k])
        #     p[maxPop, k] = 0.0
        #     p[maxPop, k] = nTotal[k] - np.sum(p[:, k])


@njit
def time_dep_update_impl(theta, dt, Gamma, GammaPrev, n, nPrev):
    Nlevel = n.shape[0]
    Nspace = n.shape[1]

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
        nk[:] = (1.0 - theta) * dt * GammaPrev[:,:, k] @ nPrevIter + nPrevIter

        nNew = np.linalg.solve(Gam, nk)
        n[:, k] = nNew
        atomDelta = max(atomDelta, np.nanmax(np.abs(1.0 - nCurrent / nNew)))

    return atomDelta

class MsLightweaverManager:

    def __init__(self, atmost, outputDir,
                 atoms, activeAtoms=['H', 'Ca'],
                 startingCtx=None, conserveCharge=False,
                 populationTransportMode='Advect',
                 prd=False):
        # check_write_git_revision(outputDir)
        self.atmost = atmost
        self.outputDir = outputDir
        self.conserveCharge = conserveCharge
        self.at = get_global_atomic_table()
        self.idx = 0
        self.nHTot = atmost.d1 / (self.at.weightPerH * Const.Amu)
        self.prd = prd
        if populationTransportMode == 'Advect':
            self.advectPops = True
            self.rescalePops = False
        elif populationTransportMode == 'Rescale':
            self.advectPops = False
            self.rescalePops = True
        elif populationTransportMode is None or populationTransportMode == 'None':
            self.advectPops = False
            self.rescalePops = False
        else:
            raise ValueError('Unknown populationTransportMode: %s' % populationTransportMode)

        if startingCtx is not None:
            self.ctx = startingCtx
            args = startingCtx.arguments
            self.atmos = args['atmos']
            self.spect = args['spect']
            self.aSet = self.spect.radSet
            self.eqPops = args['eqPops']
        else:
            nHTot = self.nHTot[0]
            self.atmos = Atmosphere(scale=ScaleType.Geometric, depthScale=np.copy(atmost.z1[0]), temperature=np.copy(atmost.tg1[0]), vlos=np.copy(atmost.vz1[0]), vturb=np.copy(atmost.vturb), ne=np.copy(atmost.ne1[0]), nHTot=nHTot)

            self.atmos.convert_scales()
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

            self.ctx = lw.Context(self.atmos, self.spect, self.eqPops, initSol=InitialSolution.Lte, conserveCharge=self.conserveCharge, Nthreads=12)

        self.atmos.bHeat = np.ones_like(self.atmost.bheat1[0]) * 1e-20
        self.atmos.hPops = self.eqPops['H']
        np.save(self.outputDir + 'Wavelength.npy', self.ctx.spect.wavelength)

        # NOTE(cmo): Set up background
        # self.opc = OpcFile('opctab_cmo_mslw.dat')
        # # self.opc = OpcFile()
        # opcWvl = self.opc.wavel
        # self.opcWvl = opcWvl
        # # NOTE(cmo): Find mapping from wavelength array to opctab array, with
        # # constant background over the region of each line. Are overlaps a
        # # problem here? Probably -- but let's see the spectrum in practice
        # # The record to be used is the one in self.wvlIdxs + 4 due to the data
        # # layout in the opctab
        # self.wvlIdxs = np.ones_like(self.spect.wavelength, dtype=np.int64) * -1
        # lineCores = []
        # for a in self.aSet.activeSet:
        #     for l in a.lines:
        #         lineCores.append(l.lambda0 * 10)
        # lineCores = np.array(lineCores)
        # lineCoreIdxs = np.zeros_like(lineCores)
        # for i, l in enumerate(lineCores):
        #     closestIdx = np.argmin(np.abs(opcWvl - l))
        #     lineCoreIdxs[i] = closestIdx

        # for a in self.aSet.activeSet:
        #     for l in a.lines:
        #         # closestIdx = np.argmin((opcWvl - l.lambda0*10)**2)
        #         closestCore = np.argmin(np.abs((l.wavelength * 10)[:, None] - lineCores), axis=1)
        #         closestIdx = lineCoreIdxs[closestCore]
        #         sub = find_subarray(self.spect.wavelength, l.wavelength)
        #         self.wvlIdxs[sub:sub + l.wavelength.shape[0]] = closestIdx
        # for i, v in enumerate(self.wvlIdxs):
        #     if v >= 0:
        #         continue

        #     closestIdx = np.argmin(np.abs(opcWvl - self.spect.wavelength[i]*10))
        #     self.wvlIdxs[i] = closestIdx
        # self.opctabIdxs = self.wvlIdxs + 4

        # NOTE(cmo): Compute initial background opacity
        # np.save('chi.npy', self.ctx.background.chi)
        # np.save('eta.npy', self.ctx.background.eta)
        # np.save('sca.npy', self.ctx.background.sca)
        # self.opac_background()

    def opac_background_wvl(self, wvlIdx):
        xlamb = self.opcWvl[wvlIdx]
        # NOTE(cmo): Don't want NLTE hydrogen pops in here, Lw handles them in-place, so just take LTE
        ne1 = self.atmos.ne / 1e6
        tg1 = self.atmos.temperature
        hPops = (self.atmos.hPops / 1e6).T
        nHTot = self.atmos.nHTot / 1e6
        _, xcont = hydrogen_absorption(xlamb, -1, tg1, ne1, hPops, explicitLevels=False)

        # NOTE(cmo): Scattering
        xlimit = 1026.0
        xray = max(xlamb, xlimit)
        w2 = 1.0 / xray**2
        w4 = w2**2

        scatrh = w4 * (5.799e-13 + w2 * (1.422e-6 + w2 * 2.784))
        scne0 = 6.655e-25

        op = self.opc.roptab(tg1, ne1, self.opctabIdxs[wvlIdx])
        absk = op['v']
        if wvlIdx == 114:
            np.save('nHTot.npy', nHTot)
            np.save('absk.npy', absk)
        # absk = np.zeros_like(xcont)

        Bv = planck(tg1, xlamb / 10.0)

        scatne = scne0 * ne1
        xcont += absk * nHTot
        sca = scatne + scatrh * hPops[:, 0]
        # sca = 0.0
        xcont *= 1e2
        sca *= 1e2

        etacont = xcont * Bv

        # NOTE(cmo): Add scattering to final opacity.
        xcont += sca

        return etacont, xcont, sca

    def opac_background(self):
        # NOTE(cmo): Make sure table is init'ed
        self.opc.roptab(self.atmos.temperature, self.atmos.ne/1e6, 4)
        for i in range(self.opcWvl.shape[0]):
            eta, chi, sca = self.opac_background_wvl(i)
            idxs = np.argwhere(self.wvlIdxs == i).squeeze()
            self.ctx.background.eta[idxs, :] = eta
            self.ctx.background.chi[idxs, :] = chi
            self.ctx.background.sca[idxs, :] = sca

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

    def advect_pops(self):
        if self.rescalePops:
            adv = self.atmost.d1[self.idx+1] / self.atmost.d1[self.idx]
            neAdv = self.atmos.ne * adv
            self.atmos.ne[:] = neAdv
            for atom in self.aSet.activeAtoms:
                p = self.eqPops[atom.name]
                for i in range(p.shape[0]):
                    pAdv = p[i] * adv
                    p[i, :] = pAdv
        elif self.advectPops:
            # z0 = self.atmost.z1[self.idx]
            # z1 = self.atmost.z1[self.idx+1]
            # d0 = self.atmost.d1[self.idx]
            # d1 = self.atmost.d1[self.idx+1]
            # vz0 = self.atmost.vz1[self.idx]
            # vz1 = self.atmost.vz1[self.idx+1]
            # dt = self.atmost.dt[self.idx+1]

            # z0m = z0 + 0.5 * vz0 * dt
            # z0Tracer = z0 + interp1d(z1, vz1, kind=3, fill_value='extrapolate')(z0m) * dt

            # densityAdv = d1 / d0
            # for atom in self.aSet.activeAtoms:
            #     p = self.eqPops[atom.name]
            #     for i in range(p.shape[0]):
            #         p0 = p[i, :]
            #         p[i, :] = 10**(interp1d(z0Tracer, np.log10(p0), kind=3, fill_value='extrapolate')(z1))
            #     nTotal = self.eqPops.atomicPops[atom.name].nTotal
            #     nTotalAdv = nTotal * densityAdv
            #     p *= nTotalAdv / p.sum(axis=0)

            advect(self.atmost, self.idx, self.eqPops, [a.name for a in self.aSet.activeAtoms], self.at)

            # NOTE(cmo): Guess advected n_e. Will be corrected to be self
            # consistent later (in update_deps if conserveCharge=True). If
            # conserveCharge isn't true then we're using loaded n_e anyway
            # neAdv = interp1d(z0Tracer, np.log10(self.atmos.ne), kind=3, fill_value='extrapolate')(z1)
            # self.atmos.ne[:] = 10**neAdv


    def save_timestep(self):
        i = self.idx
        with open(self.outputDir + 'Step_%.6d.pickle' % i, 'wb') as pkl:
            eqPops = distill_pops(self.eqPops)
            Iwave = self.ctx.spect.I
            pickle.dump({'eqPops': eqPops, 'Iwave': Iwave, 'ne': self.atmos.ne}, pkl)

    def load_timestep(self, stepNum):
        with open(self.outputDir + 'Step_%.6d.pickle' % stepNum, 'rb') as pkl:
            step = pickle.load(pkl)

        self.idx = stepNum
        self.atmos.temperature[:] = self.atmost.tg1[self.idx]
        self.atmos.vlos[:] = self.atmost.vz1[self.idx]
        if not self.conserveCharge:
            self.atmos.ne[:] = self.atmost.ne1[self.idx]

        if self.advectPops or self.rescalePops:
            self.atmos.nHTot[:] = self.nHTot[self.idx]
        self.atmos.bHeat[:] = self.atmost.bheat1[self.idx]

        self.atmos.height[:] = self.atmost.z1[self.idx]

        for name, pops in step['eqPops'].items():
            if pops['n'] is not None:
                self.eqPops.atomicPops[name].pops[:] = pops['n']
            self.eqPops.atomicPops[name].nStar[:] = pops['nStar']
        self.atmos.ne[:] = step['ne']
        self.ctx.update_deps()

    def increment_step(self):
        self.advect_pops()
        self.idx += 1
        self.atmos.temperature[:] = self.atmost.tg1[self.idx]
        self.atmos.vlos[:] = self.atmost.vz1[self.idx]
        if not self.conserveCharge:
            self.atmos.ne[:] = self.atmost.ne1[self.idx]

        if self.advectPops or self.rescalePops:
            self.atmos.nHTot[:] = self.nHTot[self.idx]
        self.atmos.bHeat[:] = self.atmost.bheat1[self.idx]

        self.atmos.height[:] = self.atmost.z1[self.idx]
        self.ctx.update_deps()
        if self.prd:
            self.ctx.configure_hprd_coeffs()
        # self.opac_background()

    def time_dep_prev_state(self):
        s = {}
        s['pops'] = [np.copy(a.n) for a in self.ctx.activeAtoms]
        s['Gamma'] = [np.copy(a.Gamma) for a in self.ctx.activeAtoms]
        return s

    def time_dep_update(self, dt, prevState, theta=0.5):
        atoms = self.ctx.activeAtoms
        Nspace = self.atmos.Nspace

        maxDelta = 0.0
        for i, atom in enumerate(atoms):
            atomDelta = time_dep_update_impl(theta, dt, atom.Gamma, prevState['Gamma'][i],
                                             atom.n, prevState['pops'][i])

            maxDelta = max(maxDelta, atomDelta)
            s = '    %s delta = %6.4e' % (atom.atomicModel.name, atomDelta)
            print(s)

        return maxDelta

    def time_dep_step(self, nSubSteps=200, popsTol=1e-3, JTol=3e-3, theta=0.5):
        dt = self.atmost.dt[self.idx+1]
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

        prevState = self.time_dep_prev_state()
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

            if delta > 5e-1:
                continue
            if not underTol:
                underTol = True
                if sub > 0 and self.conserveCharge:
                    self.eqPops.update_lte_atoms_Hmin_pops(self.atmos, True, True)
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

    def rf_k(self, step, dt, pertSize, k):
        self.load_timestep(step)
        self.ctx.clear_ng()
        self.ctx.spect.J[:] = 0.0

        self.atmos.temperature[k] += 0.5 * pertSize
        self.ctx.update_deps()

        self.time_dep_step(popsTol=1e-3, JTol=5e-3, dt=dt)
        plus = np.copy(self.ctx.spect.I[:, -1])

        self.load_timestep(step)
        self.ctx.clear_ng()
        self.ctx.spect.J[:] = 0.0

        self.atmos.temperature[k] -= 0.5 * pertSize
        self.ctx.update_deps()

        self.time_dep_step(popsTol=1e-3, JTol=5e-3, dt=dt)
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
        d[atom.name] = convert_atomic_pops(atom)
    return d
