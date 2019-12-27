import numpy as np
from dataclasses import dataclass

@dataclass
class Atmost:
    grav: float
    tau2: float
    vturb: np.ndarray

    time: np.ndarray
    dt: np.ndarray
    z1: np.ndarray
    d1: np.ndarray
    ne1: np.ndarray
    tg1: np.ndarray
    vz1: np.ndarray
    nh1: np.ndarray

    cgs: bool = True

    def to_SI(self):
        if not self.cgs:
            return

        self.vturb /= 1e2
        self.z1 /= 1e2
        self.d1 *= 1e3
        self.ne1 *= 1e6
        self.vz1 /= 1e2
        self.nh1 *= 1e6

        self.cgs = False


def read_atmost(filename='atmost.dat') -> Atmost:
    with open(filename, 'rb') as f:
        # Record: itype 4, isize 4, cname 8 : 16
        _ = np.fromfile(f, np.int32, 1)
        itype = np.fromfile(f, np.int32, 1)
        isize = np.fromfile(f, np.int32, 1)
        cname = np.fromfile(f, 'c', 8)
        _ = np.fromfile(f, np.int32, 1)

        # Record: ntime 4, ndep 4 : 8
        _ = np.fromfile(f, np.int32, 1)
        ntime = np.fromfile(f, np.int32, 1)
        ndep = np.fromfile(f, np.int32, 1)
        _ = np.fromfile(f, np.int32, 1)

        # Record: itype 4, isize 4, cname 8 : 16
        _ = np.fromfile(f, np.int32, 1)
        itype = np.fromfile(f, np.int32, 1)
        isize = np.fromfile(f, np.int32, 1)
        cname = np.fromfile(f, 'c', 8)
        _ = np.fromfile(f, np.int32, 1)

        # Record: grav 8, tau(2) 8, vturb 8 x ndep(300) : 2416
        _ = np.fromfile(f, np.int32, 1)
        grav = np.fromfile(f, np.float64, 1)
        tau2 = np.fromfile(f, np.float64, 1)
        vturb = np.fromfile(f, np.float64, ndep[0])
        _ = np.fromfile(f, np.int32, 1)
        if grav[0] == 0.0:
            grav[0] = 10**4.44

        times = []
        dtns = []
        z1t = []
        d1t = []
        ne1t = []
        tg1t = []
        vz1t = []
        nh1t = []
        while True:
            # Record: itype 4, isize 4, cname 8 : 16
            _ = np.fromfile(f, np.int32, 1)
            itype = np.fromfile(f, np.int32, 1)
            isize = np.fromfile(f, np.int32, 1)
            cname = np.fromfile(f, 'c', 8)
            _ = np.fromfile(f, np.int32, 1)

            # Record: timep 8, dtnp 8, z1 8 * ndep(300), 
            # d1 8 * ndep(300), ne1 8 * ndep(300), 
            # tg1 8 * ndep(300), vz1 8 * ndep(300),
            # nh1 8 * 6 * ndep(300): 26416
            _ = np.fromfile(f, np.int32, 1)
            times.append(np.fromfile(f, np.float64, 1))
            if times[-1].shape != (1,):
                times.pop()
                break
            dtns.append(np.fromfile(f, np.float64, 1))
            z1t.append(np.fromfile(f, np.float64, ndep[0]))
            d1t.append(np.fromfile(f, np.float64, ndep[0]))
            ne1t.append(np.fromfile(f, np.float64, ndep[0]))
            tg1t.append(np.fromfile(f, np.float64, ndep[0]))
            vz1t.append(np.fromfile(f, np.float64, ndep[0]))
            nh1t.append(np.fromfile(f, np.float64, ndep[0] * 6).reshape(6, ndep[0]))
            _ = np.fromfile(f, np.int32, 1)

    times = np.array(times).squeeze()
    dtns = np.array(dtns).squeeze()
    z1t = np.array(z1t).squeeze()
    d1t = np.array(d1t).squeeze()
    ne1t = np.array(ne1t).squeeze()
    tg1t = np.array(tg1t).squeeze()
    vz1t = np.array(vz1t).squeeze()
    nh1t = np.array(nh1t).squeeze()

    return Atmost(grav.item(), tau2.item(), vturb, times, dtns, z1t, d1t, ne1t, tg1t, vz1t, nh1t)

