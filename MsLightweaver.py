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

OutputDir = 'TimestepsHeightTheta55/'
Path(OutputDir).mkdir(parents=True, exist_ok=True)
Path(OutputDir + '/Rfs').mkdir(parents=True, exist_ok=True)
Path(OutputDir + '/ContFn').mkdir(parents=True, exist_ok=True)
NumInterfaces = 1024
NasaAtoms = [H_6_nasa(), CaII_nasa(), He_9(), C_atom(), O_atom(), Si_atom(), Fe_atom(), 
             MgII_atom(), N_atom(), Na_atom(), S_atom()]
FchromaAtoms = [H_6(), CaII(), He_9(), C_atom(), O_atom(), Si_atom(), Fe_atom(), 
                MgII_atom(), N_atom(), Na_atom(), S_atom()]
DoAdvection = False

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

from HydroWeno.Simulation import Grid
from HydroWeno.BCs import zero_grad_bc
from HydroWeno.Weno import reconstruct_weno_nm_z
from MsLightweaverAdvector import MsLightweaverAdvector
from scipy.interpolate import interp1d

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

    def __init__(self, atmost, numInterfaces, startingCtx=None):
        self.atmost = atmost
        self.at = get_global_atomic_table()
        self.idx = 0
        self.nHTot = atmost['d1'] / (self.at.weightPerH * Const.Amu)
        self.numInterfaces = numInterfaces

        if startingCtx is not None:
            self.ctx = startingCtx
            args = startingCtx.arguments
            self.atmos = args['atmos']
            self.spect = args['spect']
            self.aSet = self.spect.radSet
            self.eqPops = args['eqPops']
        else:
            cmassGrid = 'cmassGrid' in atmost

            if DoAdvection and cmassGrid:
                raise ValueError('Currenttly not supporting cmassGrid and advection')

            nHTot = atmost['d1'][0] / (self.at.weightPerH * Const.Amu)
            if cmassGrid:
                self.atmos = Atmosphere(scale=ScaleType.ColumnMass, depthScale=np.copy(atmost['cmassGrid']), temperature=np.copy(atmost['tg1'][0]), vlos=np.copy(atmost['vz1'][0]), vturb=np.copy(atmost['vturb']), ne=np.copy(atmost['ne1'][0]), nHTot=nHTot)
            else:
                self.atmos = Atmosphere(scale=ScaleType.Geometric, depthScale=np.copy(atmost['zGrid']), temperature=np.copy(atmost['tg1'][0]), vlos=np.copy(atmost['vz1'][0]), vturb=np.copy(atmost['vturb']), ne=np.copy(atmost['ne1'][0]), nHTot=nHTot)

            self.atmos.convert_scales()
            self.atmos.quadrature(5)

            self.aSet = RadiativeSet(FchromaAtoms)
            self.aSet.set_active('H', 'Ca')
            # NOTE(cmo): Radyn seems to compute the collisional rates once per
            # timestep(?) and we seem to get a much better agreement for Ca
            # with the CH rates when H is set to LTE for the initial timestep.
            # Might be a bug in my implementation though.

            self.spect = self.aSet.compute_wavelength_grid()

            self.mols = MolecularTable()
            self.eqPops = self.aSet.compute_eq_pops(self.atmos, self.mols)
            # self.eqPops = self.aSet.iterate_lte_ne_eq_pops(self.mols, self.atmos)

            # self.ctx = LwContext(self.atmos, self.spect, self.eqPops, initSol=InitialSolution.Lte, conserveCharge=False, Nthreads=12, ngOptions=NgOptions(3, 7, 50))
            self.ctx = LwContext(self.atmos, self.spect, self.eqPops, initSol=InitialSolution.Lte, conserveCharge=False, Nthreads=12)

        self.atmos.bHeat = np.ones_like(self.atmost['bheat1'][0]) * 1e-20
        self.atmos.hPops = self.eqPops['H']

        if DoAdvection:
            numPopRows = 0
            for atom in self.aSet.activeAtoms:
                numPopRows += self.eqPops[atom.name].shape[0]

            try:
                self.atmos.originalHeight
            except AttributeError:
                self.atmos.originalHeight = np.copy(self.atmos.height)
            
            # Expand grid slightly with linear extrapolation, so cmass interfaces are preserved better for interpolation back
            staticHeightInterfaces = interp1d(np.linspace(0, 1, self.atmos.height.shape[0]),
                                            self.atmos.originalHeight[::-1], 
                                            fill_value='extrapolate')(np.linspace(-0.02,
                                                                                    1.01, 
                                                                                    numInterfaces))
            self.staticGrid = Grid(staticHeightInterfaces, 4)
            self.adv = MsLightweaverAdvector(self.staticGrid, self.atmos.originalHeight, 
                                            self.atmos.cmass, self.atmost,
                                            simple_advection_bcs(), numPopRows=numPopRows)
            # Make height and density grid self-consistent with hydro. Slight offsets appear due to cell-centres vs interfaces
            self.atmos.height[:] = self.adv.height_from_cmass(self.atmos.cmass)
            newRho = self.adv.rho_from_cmass(self.atmos.cmass)
            self.atmos.nHTot[:] = newRho / (self.at.weightPerH*Const.Amu)
            self.eqPops.update_lte_atoms_Hmin_pops(self.atmos)
            self.JPrev = np.copy(self.ctx.spect.J)

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

    def initial_stat_eq(self, nScatter=3, nMaxIter=1000, popTol=1e-3, JTol=3e-3):
        for i in range(nMaxIter):
            dJ = self.ctx.formal_sol_gamma_matrices()
            if i < nScatter:
                continue

            delta = self.ctx.stat_equil()

            if self.ctx.crswDone and dJ < JTol and delta < popTol:
                print('Stat eq converged in %d iterations' % (i+1))
                break
        else:
            raise ConvergenceError('Stat Eq did not converge.')

    def update_height(self):
        if self.atmos.scale == lw.ScaleType.Geometric:
            return

        height = self.atmos.height
        cmass = self.atmos.cmass
        rhoSI = self.atmost['d1'][self.idx]
        height[0] = 0.0
        for k in range(1, cmass.shape[0]):
            height[k] = height[k-1] - 2.0 * (cmass[k] - cmass[k-1]) / (rhoSI[k-1] + rhoSI[k])

        # NOTE(cmo): ish.
        # NOTE(cmo): What I mean by ish, is this assumes that tau500 doesn't change, which it clearly does, but we never refer to tau500 in the computation or save it, so it currently doesn't matter.
        hTau1 = np.interp(1.0, self.atmos.tau_ref, height)
        height -= hTau1

    def advect_pops(self):
        activePops = []
        popSizes = []
        for atom in self.aSet.activeAtoms:
            p = self.eqPops[atom.name]
            p = p / np.sum(p, axis=0)
            activePops.append(p)
            popSizes.append(p.shape[0])
        pops = np.concatenate(activePops, axis=0)

        self.adv.fill_from_pops(self.atmos.cmass, pops, popSizes)
        self.adv.step()

        newPops = self.adv.interp_to_cmass(self.atmos.cmass)
        newHeight = self.adv.height_from_cmass(self.atmos.cmass)
        newRho = self.adv.rho_from_cmass(self.atmos.cmass)

        self.atmos.height[:] = newHeight
        self.atmos.nHTot[:] = newRho / (self.at.weightPerH*Const.Amu)

        takeFrom = 0
        for atom in self.aSet.activeAtoms:
            atomPop = self.eqPops[atom.name]
            nLevels = atomPop.shape[0]
            p = newPops[takeFrom:takeFrom+nLevels, :]
            p = p / np.sum(p, axis=0)
            atomPop[:] = p * self.atmos.nHTot * self.at[atom.name].abundance
            takeFrom += nLevels

    def load_timestep(self, stepNum):
        with open(OutputDir + 'Step_%.6d.pickle' % stepNum, 'rb') as pkl:
            step = pickle.load(pkl)

        for name, pops in step['eqPops'].items():
            if pops['n'] is not None:
                self.eqPops.atomicPops[name].pops[:] = pops['n']
            self.eqPops.atomicPops[name].nStar[:] = pops['nStar']

        self.idx = stepNum - 1
        self.increment_step()

    def increment_step(self):
        if DoAdvection:
            self.advect_pops()
        self.idx += 1
        self.atmos.temperature[:] = self.atmost['tg1'][self.idx]
        self.atmos.vlos[:] = self.atmost['vz1'][self.idx]
        self.atmos.ne[:] = self.atmost['ne1'][self.idx]
        # Now done with the advection
        if False and not DoAdvection:
            self.atmos.nHTot[:] = self.nHTot[self.idx]
        self.atmos.bHeat[:] = self.atmost['bheat1'][self.idx]

        # self.atmos.convert_scales()
        # NOTE(cmo): Convert scales is slow due to recomputing the tau500 opacity (pure python EOS), we only really need the height at the moment. So let's just do that quickly here.
        if not DoAdvection:
            self.update_height()
        # NOTE(cmo): Yes, for now this is also recomputing the background... it'll be fine though
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
        dt = self.atmost['dt'][self.idx+1]
        # self.ctx.spect.J[:] = 0.0

        prevState = self.time_dep_prev_state()
        for sub in range(nSubSteps):
            # self.JPrev[:] = self.ctx.spect.J

            dJ = self.ctx.formal_sol_gamma_matrices()
            # if sub > 2:
            # delta, prevState = self.ctx.time_dep_update(dt, prevState)
            delta = self.time_dep_update(dt, prevState, theta=theta)

            if sub > 1 and delta < popsTol and dJ < JTol:
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
            with open(OutputDir + 'Fails.txt', 'a') as f:
                f.write('%d, %.4e %.4e\n' % (self.idx, delta, dJ))

            with open(OutputDir + 'NonConvergenceData_%.6d.pickle' % (self.idx), 'wb') as pkl:
                pickle.dump(sourceData, pkl)
        # self.ctx.time_dep_conserve_charge(prevState)

filesInOutDir = [f for f in os.listdir(OutputDir) if f.startswith('Step_')]
if len(filesInOutDir) > 0:
    print('Timesteps already present in output directory (%s), proceed? [Y/n]' % OutputDir)
    inp = input()
    if len(inp) > 0 and inp[0].lower() == 'n':
        raise ValueError('Data in output directory')

with open('RadynData.pickle', 'rb') as pkl:
    atmost = pickle.load(pkl)

if path.isfile(OutputDir + 'StartingContext.pickle'):
    with open(OutputDir + 'StartingContext.pickle', 'rb') as pkl:
        startingCtx = pickle.load(pkl)
else:
    startingCtx = None

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

def save_timestep(i):
    with open(OutputDir + 'Step_%.6d.pickle' % i, 'wb') as pkl:
        eqPops = distill_pops(ms.eqPops)
        Iwave = ms.ctx.spect.I
        pickle.dump({'eqPops': eqPops, 'Iwave': Iwave}, pkl)

start = time.time()
ms = MsLightweaverManager(atmost, NumInterfaces, startingCtx=startingCtx)
ms.initial_stat_eq(popTol=1e-3)
save_timestep(0)
np.save(OutputDir + 'Wavelength.npy', ms.ctx.spect.wavelength)

# for i in range(10):
#     ms.opac_background()
#     ms.initial_stat_eq()
# save_timestep(0)
# np.save('chiOpac.npy', ms.ctx.background.chi)
# np.save('etaOpac.npy', ms.ctx.background.eta)
# np.save('scaOpac.npy', ms.ctx.background.sca)
# raise ValueError

if startingCtx is None:
    with open(OutputDir + 'StartingContext.pickle', 'wb') as pkl:
        pickle.dump(ms.ctx, pkl)

maxSteps = ms.atmost['time'].shape[0] - 1
ms.atmos.bHeat[:] = ms.atmost['bheat1'][0]
firstStep = 0
# firstStep = 7685
if firstStep != 0:
    ms.load_timestep(firstStep)
    ms.ctx.spect.J[:] = 0.0
    ms.ctx.formal_sol_gamma_matrices()

for i in range(firstStep, maxSteps):
    stepStart = time.time()
    if i != 0:
        ms.increment_step()
    ms.time_dep_step(popsTol=1e-3, JTol=5e-3, nSubSteps=1000, theta=0.55)
    ms.ctx.clear_ng()
    save_timestep(i+1)
    stepEnd = time.time()
    print('-------')
    print('Timestep %d done (%f s)' % ((i+1), ms.atmost['time'][i+1]))
    print('Time taken for step %.2e s' % (stepEnd - stepStart))
    print('-------')
end = time.time()
print('Total time taken %.4e' % (end - start))

notify = Notify()
notify.read_config()
notify.send('MsLightweaver done!')
