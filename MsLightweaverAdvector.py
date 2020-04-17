import numpy as np
from HydroWeno.Advector import Advector, advection_flux
from HydroWeno.Simulation import Grid
from HydroWeno.Weno import reconstruct_weno_nm_z, reconstruct_weno_z
from scipy.interpolate import interp1d

def cfl(grid, data):
    vel = np.abs(data[0])
    dt = 0.8 * np.min(grid.dx / vel)
    return dt

class MsLightweaverAdvectorCmass:
    def __init__(self, grid, initialHeight, refCmassGrid, atmost, bcs, numPopRows=0):
        self.grid = grid
        self.atmost = atmost
        self.idx = 0
        self.refCmass = refCmassGrid

        startData = np.zeros((numPopRows+2, grid.griMax))
        height = self.grid.cc[self.grid.griBeg:self.grid.griEnd]
        startData[1, :] = interp1d(initialHeight[::-1], atmost['d1'][0, ::-1], bounds_error=False, fill_value=(atmost['d1'][0, -1], atmost['d1'][0, 0]))(grid.cc)

        self.ad = Advector(grid, startData, bcs, reconstruct_weno_nm_z)

    def fill_from_pops(self, cmassGrid, pops, popDims):
        cmass = self.cmass
        data = self.data[:, self.grid.griBeg:self.grid.griEnd]
        for i in range(pops.shape[0]):
            data[i+2, :] = interp1d(cmassGrid, pops[i], kind=1, fill_value=(pops[i][0], pops[i][-1]), bounds_error=False)(cmass)
            
        start = 2
        for p in popDims:
            data[start:start+p, :] /= np.sum(data[start:start+p, :], axis=0)
            start += p

    def interp_to_cmass(self, cmassGrid):
        pops = np.zeros((self.data.shape[0]-2, cmassGrid.shape[0]))
        cmass = self.cmass
        data = self.data[:, self.grid.griBeg:self.grid.griEnd]
        for i in range(pops.shape[0]):
            pops[i] = interp1d(cmass, data[i+2], bounds_error=False, fill_value=(data[i+2, -1], data[i+2, 0]))(cmassGrid)
        return pops

    def height_from_cmass(self, cmassGrid):
        cmass = self.cmass
        grid = self.grid.cc[self.grid.griBeg:self.grid.griEnd]
        height = interp1d(cmass, grid, bounds_error=False, fill_value=(grid[-1], grid[0]))(cmassGrid)
        return height

    def rho_from_cmass(self, cmassGrid):
        cmass = self.cmass
        grid = self.data[1, self.grid.griBeg:self.grid.griEnd]
        rho = interp1d(cmass, grid, bounds_error=False, fill_value=(grid[-1], grid[0]))(cmassGrid)
        return rho
    
    def vel(self, idx):
        cmass = self.cmass
        vel = interp1d(self.refCmass[::-1], self.atmost['vz1'][idx, ::-1], kind=3, fill_value=0.0, bounds_error=False)(cmass)
        return vel
    
    def step(self, numSubSteps=5000):
        self.ad.data[0, self.grid.griBeg:self.grid.griEnd] = self.vel(self.idx)
        dtRadyn = self.atmost['dt'][self.idx+1]
        dtMax = cfl(self.grid, self.ad.data)
        if dtRadyn > dtMax:
            subTime = 0.0
            for i in range(numSubSteps):
                dt = dtMax
                if subTime + dt > dtRadyn:
                    dt = dtRadyn - subTime
                self.ad.step(dt)
                subTime += dt
                if subTime >= dtRadyn:
                    break
            else:
                raise Exception('Convergence')
        else:
            self.ad.step(dtRadyn)
        self.idx += 1

    @property
    def data(self):
        return self.ad.data

    @property
    def cmass(self):
        grid = self.grid.interfaces[self.grid.griBeg:self.grid.griEnd+1]
        gridFlip = grid[::-1]

        rho = self.data[1, ::-1]

        cmass = np.zeros_like(grid)
        cmassFlip = cmass[::-1]
        cmassFlip[0] = 1e-9
        for k in range(1, cmass.shape[0]):
            cmassFlip[k] = cmassFlip[k-1] + np.abs(gridFlip[k-1] - gridFlip[k]) * rho[k]
        cmassCc = 0.5 * (cmass[1:] + cmass[:-1])

        return cmassCc

class MsLightweaverAdvectorHeight:
    def __init__(self, atmost, bcs, numPopRows=0):
        heightGrid = atmost['zGrid']
        heightInterfaces = np.copy(np.concatenate([
            [heightGrid[0] + 0.5 * (heightGrid[0] - heightGrid[1])],
            0.5 * (heightGrid[1:] + heightGrid[:-1]),
            [heightGrid[-1] - 0.5 * (heightGrid[-2] - heightGrid[-1])]
        ])[::-1])
        self.heightInterface = heightInterfaces
        grid = Grid(heightInterfaces, numGhost=4)
        self.grid = grid
        self.atmost = atmost
        self.idx = 0

        height = self.grid.cc[self.grid.griBeg:self.grid.griEnd]
        self.heightCc = height
        self.heightGrid = heightGrid

        startData = np.zeros((numPopRows+2, grid.griMax))
        startData[1, :] = interp1d(self.heightGrid[::-1], 
                                   atmost['d1'][0, ::-1], 
                                   bounds_error=False, 
                                   fill_value=(atmost['d1'][0, -1], 
                                               atmost['d1'][0, 0])
                                   )(grid.cc)

        self.ad = Advector(grid, startData, bcs, reconstruct_weno_nm_z)

    def fill_from_pops(self, pops, popDims):
        data = self.data[:, self.grid.griBeg:self.grid.griEnd]
        for i in range(pops.shape[0]):
            data[i+2, :] = interp1d(self.heightGrid, pops[i], 
                                    kind=1, fill_value=(pops[i][0], pops[i][-1]), bounds_error=False)(self.heightCc)
            
        start = 2
        for p in popDims:
            data[start:start+p, :] /= np.sum(data[start:start+p, :], axis=0)
            start += p

    def pops(self):
        pops = np.zeros((self.data.shape[0]-2, cmassGrid.shape[0]))
        data = self.data[:, self.grid.griBeg:self.grid.griEnd]
        for i in range(pops.shape[0]):
            pops[i] = interp1d(self.heightCc, data[i+2], 
                               bounds_error=False, 
                               fill_value=(data[i+2, -1], data[i+2, 0]))(self.heightGrid)
        return pops

    def rho(self):
        grid = self.data[1, self.grid.griBeg:self.grid.griEnd]
        rho = interp1d(self.heightCc, grid, bounds_error=False, 
                       fill_value=(grid[-1], grid[0]))(self.heightGrid)
        return rho
    
    def vel(self, idx, timeStepAdvance=0.0):
        assert timeStepAdvance < 1.0
        vel = interp1d(self.heightGrid[::-1], self.atmost['vz1'][idx, ::-1], 
                       kind=1, fill_value=0.0, bounds_error=False)(self.heightCc)
        if timeStepAdvance > 0.0:
            vel2 = interp1d(self.heightGrid[::-1], self.atmost['vz1'][idx+1, ::-1], 
                            kind=1, fill_value=0.0, bounds_error=False)(self.heightCc)
            # TODO(cmo): Maybe add some interpolation smoothing, like cosine
            vel *= (1.0 - timeStepAdvance)
            vel += vel2 * timeStepAdvance
        return vel
    
    def step(self, numSubSteps=5000):
        self.ad.data[0, self.grid.griBeg:self.grid.griEnd] = self.vel(self.idx)
        dtRadyn = self.atmost['dt'][self.idx+1]
        dtMax = cfl(self.grid, self.ad.data)
        if dtRadyn > dtMax:
            subTime = 0.0
            for i in range(numSubSteps):
                dt = dtMax
                if subTime + dt > dtRadyn:
                    dt = dtRadyn - subTime
                self.ad.step(dt)
                subTime += dt
                if subTime >= dtRadyn:
                    break
                timestepFrac =  subTime / dtRadyn
                advancedVel = self.ad.data[0, self.grid.griBeg:self.grid.griEnd]
                advancedVel[:] = self.vel(self.idx, timeStepAdvance=timestepFrac)
                dtMax = cfl(self.grid, self.ad.data)
            else:
                raise Exception('Convergence')
        else:
            self.ad.step(dtRadyn)
        self.idx += 1

    @property
    def data(self):
        return self.ad.data