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
from HydroWeno.Simulation import Grid
from HydroWeno.BCs import zero_grad_bc
from HydroWeno.Weno import reconstruct_weno_nm_z
from scipy.interpolate import interp1d

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


def simple_advection_bcs():
    lower_bc = zero_grad_bc('Lower')
    upper_bc = zero_grad_bc('Upper')
    def apply_bcs(grid, V):
        lower_bc(grid, V)
        upper_bc(grid, V)

    return apply_bcs

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
                 startingCtx=None, conserveCharge=False):
        check_write_git_revision(outputDir)
        self.atmost = atmost
        self.outputDir = outputDir
        self.conserveCharge = conserveCharge
        self.at = get_global_atomic_table()
        self.idx = 0
        self.nHTot = atmost.d1 / (self.at.weightPerH * Const.Amu)

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

            self.ctx = lw.Context(self.atmos, self.spect, self.eqPops, initSol=InitialSolution.Lte, conserveCharge=False, Nthreads=12)

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
        for i in range(NmaxIter):
            dJ = self.ctx.formal_sol_gamma_matrices()
            if i < Nscatter:
                continue

            delta = self.ctx.stat_equil()
            if self.conserveCharge:
                self.ctx.nr_post_update()
                for p in self.eqPops.atomicPops:
                    p.nStar[:] = lw.lte_pops(p.model, self.atmos.temperature, self.atmos.ne, p.nTotal)

            if self.ctx.crswDone and dJ < JTol and delta < popTol:
                print('Stat eq converged in %d iterations' % (i+1))
                break
        else:
            raise ConvergenceError('Stat Eq did not converge.')

    def advect_pops(self):
        adv = self.atmost.d1[self.idx+1] / self.atmost.d1[self.idx]
        neAdv = self.atmos.ne * adv
        self.atmos.ne[:] = neAdv
        for atom in self.aSet.activeAtoms:
            p = self.eqPops[atom.name]
            for i in range(p.shape[0]):
                pAdv = p[i] * adv
                p[i, :] = pAdv

    def save_timestep(self):
        i = self.idx
        with open(self.outputDir + 'Step_%.6d.pickle' % i, 'wb') as pkl:
            eqPops = distill_pops(self.eqPops)
            Iwave = self.ctx.spect.I
            pickle.dump({'eqPops': eqPops, 'Iwave': Iwave, 'ne': self.atmos.ne}, pkl)

    def load_timestep(self, stepNum):
        with open(self.outputDir + 'Step_%.6d.pickle' % stepNum, 'rb') as pkl:
            step = pickle.load(pkl)

        for name, pops in step['eqPops'].items():
            if pops['n'] is not None:
                self.eqPops.atomicPops[name].pops[:] = pops['n']
            self.eqPops.atomicPops[name].nStar[:] = pops['nStar']

        self.idx = stepNum - 1
        self.increment_step()

    def increment_step(self):
        self.advect_pops()
        self.idx += 1
        self.atmos.temperature[:] = self.atmost.tg1[self.idx]
        self.atmos.vlos[:] = self.atmost.vz1[self.idx]
        if not self.conserveCharge:
            self.atmos.ne[:] = self.atmost.ne1[self.idx]

        self.atmos.nHTot[:] = self.nHTot[self.idx]
        self.atmos.bHeat[:] = self.atmost.bheat1[self.idx]

        self.atmos.height[:] = self.atmost.z1[self.idx]
        self.ctx.update_deps()
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
        # self.ctx.spect.J[:] = 0.0

        prevState = self.time_dep_prev_state()
        for sub in range(nSubSteps):
            # self.JPrev[:] = self.ctx.spect.J

            dJ = self.ctx.formal_sol_gamma_matrices()
            # if sub > 2:
            # delta, prevState = self.ctx.time_dep_update(dt, prevState)
            delta = self.time_dep_update(dt, prevState, theta=theta)
            if self.conserveCharge:
                dNrPops = self.ctx.nr_post_update(timeDependentData={'dt': dt, 'nPrev': prevState['pops']})
                for p in self.eqPops.atomicPops:
                    p.nStar[:] = lw.lte_pops(p.model, self.atmos.temperature, self.atmos.ne, p.nTotal)

            if sub > 1 and delta < popsTol and dJ < JTol and dNrPops < popsTol:
                break
        else:
            self.ctx.depthData.fill = True
            self.ctx.formal_sol_gamma_matrices()
            self.ctx.depthData.fill = False
            
            sourceData = {'chi': np.copy(self.ctx.depthData.chi),
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
