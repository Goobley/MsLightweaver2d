import lightweaver as lw
import numpy as np
import zarr
from scipy.signal import wiener
from weno4 import weno4

def interp(xs, xp, fp):
    return weno4(xs, xp, fp)

def FastBackground(*args):
    import lightweaver.LwCompiled
    return lightweaver.LwCompiled.FastBackground(*args, Nthreads=72)

def initial_1d_ctx(zAxis, atmost, atoms, activeAtoms, nHTot,
                   conserveCharge=False, Nthreads=16):
    temperature = weno4(zAxis, atmost.z1[0], atmost.tg1[0])
    vlos = weno4(zAxis, atmost.z1[0], atmost.vz1[0])
    vturb = np.ones_like(vlos) * 2e3
    ne1 = weno4(zAxis, atmost.z1[0], atmost.ne1[0])
    nHTot = weno4(zAxis, atmost.z1[0], nHTot[0])
    atmos = lw.Atmosphere.make_1d(scale=lw.ScaleType.Geometric,
                                  depthScale=zAxis,
                                  temperature=temperature,
                                  vlos=vlos, vturb=vturb,
                                  ne=ne1, nHTot=nHTot)
    atmos.quadrature(5)
    aSet = lw.RadiativeSet(atoms)
    aSet.set_active(*activeAtoms)
    spect = aSet.compute_wavelength_grid()

    if conserveCharge:
        eqPops = aSet.iterate_lte_ne_eq_pops(atmos)
    else:
        eqPops = aSet.compute_eq_pops(atmos)
    atmos.hPops = eqPops['H']
    atmos.bHeat = np.ones_like(temperature) * 1e-20

    ctx = lw.Context(atmos, spect, eqPops,
                     initSol=lw.InitialSolution.Lte,
                     conserveCharge=conserveCharge,
                     Nthreads=Nthreads,
                     backgroundProvider=FastBackground)
    return ctx


class MsLw2dPeriodic:
    def __init__(self, outputDir, atmost, Nz, xAxis,
                 atoms,
                 activeAtoms=['H', 'Ca'],
                 NcentralColumnsFromFlare=5,
                 maxZ=None,
                 startingCtx=None,
                 conserveCharge=False,
                 saveJ=False):

        self.atmost = atmost
        self.Nz = Nz
        self.maxZ = maxZ
        self.xAxis = xAxis
        self.conserveCharge = conserveCharge
        self.outputDir = outputDir
        self.activeAtoms = activeAtoms
        self.atoms = atoms
        self.saveJ = saveJ
        self.NcentralColumnsFromFlare = NcentralColumnsFromFlare
        self.flareColStart = self.xAxis.shape[0] // 2 - NcentralColumnsFromFlare // 2
        self.flareColEnd = self.flareColStart + NcentralColumnsFromFlare

        # NOTE(cmo): Get z-axis for first step
        self.idx = 1
        self.zAxis = self.next_z_axis(maxZ=self.maxZ)
        self.idx = 0

        zAxis = self.zAxis
        self.nHTotRadyn0 = atmost.d1[0] / (lw.DefaultAtomicAbundance.massPerH * lw.Amu)
        self.nHTot = atmost.d1 / (lw.DefaultAtomicAbundance.massPerH * lw.Amu)

        # NOTE(cmo): Set up 2D atmosphere
        Nz = self.zAxis.shape[0]
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
            temperature[...] = weno4(self.zAxis, atmost.z1[0], atmost.tg1[0])[:, None]
            ne = np.zeros((Nz, Nx))
            ne[...] = weno4(self.zAxis, atmost.z1[0], atmost.ne1[0])[:, None]
            vz = np.zeros((Nz, Nx))
            vx = np.zeros((Nz, Nx))
            vturb = np.ones((Nz, Nx)) * 2e3
            nHTot = np.zeros((Nz, Nx))
            nHTot[...] = weno4(self.zAxis, atmost.z1[0], self.nHTot[0])[:, None]
            self.atmos2d = lw.Atmosphere.make_2d(height=zAxis, x=xAxis,
                                                 temperature=temperature,
                                                 ne=ne, vx=vx, vz=vz,
                                                 vturb=vturb, nHTot=nHTot)
            self.aSet = lw.RadiativeSet(atoms)
            self.aSet.set_active(*activeAtoms)
            self.spect = self.aSet.compute_wavelength_grid()
            self.eqPops2d = self.aSet.compute_eq_pops(self.atmos2d)
            ctx1d = initial_1d_ctx(self.zAxis, self.atmost, self.atoms,
                                   self.activeAtoms, self.nHTot,
                                   conserveCharge=self.conserveCharge)
            print('-- Iterating initial context for guess populations')
            lw.iterate_ctx_se(ctx1d)
            eqPops1d = ctx1d.eqPops
            for atom in activeAtoms:
                self.eqPops2d[atom].reshape(-1, Nz, Nx)[...] = eqPops1d[atom][:, :, None]
            # if self.conserveCharge:
            #     ne[...] = ctx1d.atmos.ne[:, None]
            print('-- Copied Initial Populations')

            self.atmos2d.hPops = self.eqPops2d['H']
            self.atmos2d.bHeat = np.zeros(Nz * Nx)
            self.atmos2d.quadrature(Nquad2d)
            self.ctx = lw.Context(self.atmos2d, self.spect, self.eqPops2d,
                                  Nthreads=72,
                                  conserveCharge=conserveCharge,
                                  backgroundProvider=FastBackground)

            simParams = self.zarrStore.require_group('SimParams')
            simParams['zAxisInitial'] = np.copy(self.zAxis)
            simParams['xAxis'] = self.xAxis
            simParams['wavelength'] = self.ctx.spect.wavelength
            simParams['flareColStart'] = self.flareColStart
            simParams['flareColEnd'] = self.flareColEnd
            simParams['maxZ'] = 0.0 if self.maxZ is None else self.maxZ
            ics = self.zarrStore.require_group('InitialConditions')
            atmos = self.atmos2d.dimensioned_view()
            ics['temperature'] = atmos.temperature
            ics['vz'] = atmos.vz
            ics['vx'] = atmos.vx
            ics['vturb'] = atmos.vturb
            ics['ne'] = atmos.ne
            ics['nHTot'] = atmos.nHTot

            timeData = self.zarrStore.require_group('SimOutput')
            timeData['zAxis'] = zarr.zeros((0, Nz), chunks=(1, Nz))
            self.zGridStore = timeData['zAxis']
            self.timeRadStore = timeData.require_group('Radiation')
            self.timePopsStore = timeData.require_group('Populations')
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

    # TODO(cmo): Want to be able to limit the maxZ


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

    def load_timestep(self, stepNum, destroyLaterSteps=False):
        self.idx = stepNum

        for name, pops in self.nltePopsStore.items():
            self.eqPops2d.atomicPops[name].pops[:] = pops[self.idx]
            if destroyLaterSteps:
                # NOTE(cmo): Remove entries after the one being loaded
                pops.resize(self.idx+1, pops.shape[1], pops.shape[2])

        for name, pops in self.ltePopsStore.items():
            self.eqPops2d.atomicPops[name].nStar[:] = pops[self.idx]
            if destroyLaterSteps:
                # NOTE(cmo): Remove entries after the one being loaded
                pops.resize(self.idx+1, pops.shape[1], pops.shape[2])

        zGridStore = self.zGridStore
        self.atmos2d.z[:] = zGridStore[self.idx]
        if destroyLaterSteps:
            zGridStore.resize(self.idx+1, *zGridStore.shape[1:])

        neStore = self.neStore
        self.atmos2d.ne[:] = neStore[self.idx]
        if destroyLaterSteps:
            neStore.resize(self.idx+1, *neStore.shape[1:])

        shape = self.timeRadStore['I'].shape
        self.ctx.spect.I[:] = self.timeRadStore['I'][self.idx]
        if destroyLaterSteps:
            self.timeRadStore['I'].resize(self.idx+1, *shape[1:])

        if self.saveJ:
            shape = self.timeRadStore['J'].shape
            self.ctx.spect.J[:] = self.timeRadStore['J'][self.idx]
            if destroyLaterSteps:
                self.timeRadStore['J'].resize(self.idx+1, *shape[1:])

        # self.copy_flare_col_thermodynamics()
        self.ctx.update_deps()

    # def copy_flare_col_thermodynamics(self):
    #     for name, pops in self.nltePopsStore.items():
    #         pops1 = self.ms.eqPops[name]
    #         Nlevel = pops1.shape[0]
    #         Nz = pops1.shape[1]
    #         self.eqPops2d.atomicPops[name].pops.reshape(Nlevel,Nz,-1)[:,:,0] = pops1

    #     for name, pops in self.ltePopsStore.items():
    #         pops1 = self.ms.eqPops.atomicPops[name].nStar
    #         Nlevel = pops1.shape[0]
    #         Nz = pops1.shape[1]
    #         self.eqPops2d.atomicPops[name].nStar.reshape(Nlevel,Nz,-1)[:,:,0] = pops1

    #     self.atmos2d.ne.reshape(Nz,-1)[:, 0] = self.ms.atmos.ne[:]
    #     self.atmos2d.nHTot.reshape(Nz,-1)[:, 0] = self.ms.atmos.nHTot[:]
    #     self.atmos2d.temperature.reshape(Nz,-1)[:, 0] = self.ms.atmos.temperature[:]
    #     self.atmos2d.vz.reshape(Nz,-1)[:,0] = self.ms.atmos.vlos[:]


    def increment_step(self):
        self.idx += 1
        # NOTE(cmo): First compute new zGrid for coming step
        prevZAxis = self.zAxis
        self.zAxis = self.next_z_axis(maxZ=self.maxZ)
        zGrid = self.zAxis

        Nx = self.atmos2d.Nx

        zRadyn = self.atmost.z1[0]
        self.atmos2d.z[:] = self.zAxis
        temperature = interp(zGrid, zRadyn, self.atmost.tg1[0])
        temp2d = self.atmos2d.temperature.reshape(self.Nz, Nx)
        temp2d[...] = temperature[:, None]

        vz2d = self.atmos2d.vz.reshape(self.Nz, self.Nx)

        ne2d = self.atmos2d.ne.reshape(self.Nz, Nx)

        nHTot = interp(zGrid, zRadyn, self.nHTotRadyn0)
        nHTot2d = self.atmos2d.nHTot.reshape(self.Nz, Nx)
        nHTot2d[...] = nHTot[:, None]

        bHeat2d = self.atmos2d.bHeat.reshape(self.Nz, Nx)

        if not self.conserveCharge:
            ne = interp(zGrid, zRadyn, self.atmost.ne1[0])
            ne2d[...] = ne[:, None]
        else:
            for x in range(Nx):
                ne2d[:, x] = interp(zGrid, prevZAxis, ne2d[:, x])

        for col in range(self.flareColStart, self.flareColEnd):
            temp2d[:, col] = interp(zGrid, zRadyn, self.atmost.tg1[self.idx])
            vz2d[:, col] = interp(zGrid, zRadyn, self.atmost.vz1[self.idx])
            nHTot2d[:, col] = interp(zGrid, zRadyn, self.nHTot[self.idx])
            bHeat2d[:, col] = interp(zGrid, zRadyn, self.atmost.bheat1[self.idx])
            if not self.conserveCharge:
                ne2d[:, col] = interp(zGrid, zRadyn, self.atmost.ne1[self.idx])

        for atom in self.eqPops2d.atomicPops:
            atom.update_nTotal(self.atmos2d)
            if atom.pops is not None:
                pops2d = atom.pops.reshape(atom.pops.shape[0], self.Nz, Nx)
                for i in range(pops2d.shape[0]):
                    for x in range(pops2d.shape[2]):
                        pops2d[i, :, x] = interp(zGrid, prevZAxis, pops2d[i, :, x])
                # NOTE(cmo): We have the new nTotal from nHTot after update_deps()
                atom.pops *= (atom.nTotal / np.sum(atom.pops, axis=0))[None, :]

        self.ctx.update_deps()


    def initial_stat_eq(self, Nscatter=10, NmaxIter=1000, popsTol=1e-3):
        lw.iterate_ctx_se(self.ctx, Nscatter=Nscatter, NmaxIter=NmaxIter, popsTol=popsTol)


    def time_dep_step(self, Nsubsteps, popsTol):
        prevState = None
        dt = self.atmost.dt[self.idx+1]
        for iter2d in range(Nsubsteps):
            dJ = self.ctx.formal_sol_gamma_matrices()
            print(dJ.compact_representation())

            dPops, prevState = self.ctx.time_dep_update(dt, prevState, chunkSize=-1)
            if not self.conserveCharge:
                print(dPops.compact_representation())
                dPops = dPops.dPopsMax

            if self.conserveCharge:
                dPops = self.ctx.nr_post_update(timeDependentData={'dt': dt, 'nPrev': prevState}, chunkSize=-1)
                print(dPops.compact_representation())
                dPops = dPops.dPopsMax

            if dPops < popsTol and iter2d > 3:
                break
        else:
            raise ValueError('2D iteration failed to converge')

    def next_z_axis(self, maxZ=None):
        idxO = 0
        idxN = self.idx
        DistTol = 1
        PointTotal = self.Nz
        SmoothingSize = 15
        HalfSmoothingSize = SmoothingSize // 2

        # Merge grids
        uniqueCombined = list(np.unique(np.sort(np.concatenate((self.atmost.z1[idxO],
                                                                self.atmost.z1[idxN])))))
        if maxZ is not None:
            zMaxIdx = np.searchsorted(uniqueCombined, maxZ) + 1
            uniqueCombined = uniqueCombined[:zMaxIdx]

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
        maxZIdx = np.searchsorted(z1O, maxZ) + 1 if maxZ is not None else None
        for v in z1O[:startIdx]:
            uniqueCombined.append(v)
        for v in z1O[endIdx:maxZIdx]:
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
