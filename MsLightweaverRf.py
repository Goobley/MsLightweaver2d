import pickle
import numpy as np
import matplotlib.pyplot as plt
from lightweaver.rh_atoms import C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, Fe_atom, FeI_atom, MgII_atom, N_atom, Na_atom, S_atom
from lightweaver.atmosphere import Atmosphere, ScaleType
from lightweaver.atomic_set import RadiativeSet, SpeciesStateTable
from lightweaver.atomic_table import get_global_atomic_table
from lightweaver.molecule import MolecularTable
from lightweaver.LwCompiled import LwContext
from lightweaver.utils import InitialSolution
import lightweaver.constants as Const
from typing import List
from copy import deepcopy
from MsLightweaverAtoms import H_6, CaII, He_9
import os
import os.path as path
import time
from notify_run import Notify
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import multiprocessing
from pathlib import Path
from MsLightweaverManager import MsLightweaverManager
from MsLightweaverUtil import test_timesteps_in_dir, optional_load_starting_context

OutputDir = 'TimestepsHeightFinal/'
Path(OutputDir).mkdir(parents=True, exist_ok=True)
Path(OutputDir + '/Rfs').mkdir(parents=True, exist_ok=True)
Path(OutputDir + '/ContFn').mkdir(parents=True, exist_ok=True)

filesInOutDir = [f for f in os.listdir(OutputDir) if f.startswith('Step_')]
if len(filesInOutDir) > 0:
    print('Timesteps already present in output directory (%s), proceed? [Y/n]' % OutputDir)
    inp = input()
    if len(inp) > 0 and inp[0].lower() == 'n':
        raise ValueError('Data in output directory')

with open('RadynData.pickle', 'rb') as pkl:
    atmost = pickle.load(pkl)

startingCtx = optional_load_starting_context(OutputDir)
if startingCtx is None:
    raise ValueError('No starting context found in %s' % OutputDir)

start = time.time()
ms = MsLightweaverManager(atmost, outputDir=OutputDir, numInterfaces=None,
                          atoms=None, activeAtoms=None, startingCtx=startingCtx, 
                          doAdvection=False)
ms.initial_stat_eq()
if ms.ctx.Nthreads > 1:
    ms.ctx.Nthreads = 1

timeIdxs = np.linspace(0.5, 20, 40)
pertSize = 50
dts = [5e-4, 1e-3]
step = 545
Nspace = ms.atmos.height.shape[0]

def shush(fn, *args, **kwargs):
    with open(os.devnull, 'w') as f:
        with redirect_stdout(f):
            return fn(*args, **kwargs)

def rf_k(k, dt):
    plus, minus = ms.rf_k(step, dt, pertSize, k)
    return plus, minus

if __name__ == '__main__':
    maxCpus = min(68, multiprocessing.cpu_count())
    for t in timeIdxs:
        step = np.argwhere(ms.atmost['time'] == t).squeeze()

        contData = ms.cont_fn_data(step)
        with open(OutputDir + 'ContFn/ContFn_%d.pickle' % (step), 'wb') as pkl:
            pickle.dump(contData, pkl)

        for dt in dts:
            print('------- %d (%.2e) -------' % (step, dt))

            with ProcessPoolExecutor(max_workers=maxCpus) as exe:
                futures = [exe.submit(shush, rf_k, k, dt) for k in range(Nspace)]

                for f in tqdm(as_completed(futures), total=len(futures)):
                    pass

            rfPlus = np.zeros((Nspace, ms.ctx.spect.wavelength.shape[0]))
            rfMinus = np.zeros((Nspace, ms.ctx.spect.wavelength.shape[0]))

            for k, f in enumerate(futures):
                res = f.result()
                rfPlus[k, :] = res[0]
                rfMinus[k, :] = res[1]

            rf = (rfPlus - rfMinus) / pertSize
            with open(OutputDir + 'Rfs/Rf_temp_%.2e_%.2e_%d.pickle' % (pertSize, dt, step), 'wb') as pkl:
                pickle.dump({'rf': rf, 'pertSize': pertSize, 'dt': dt, 'step': step}, pkl)
