import numpy as np
from ReadAtmost import read_atmost
from Interp import *
import pickle


atmost = read_atmost()
atmost.to_SI()
cmass = compute_cmass(atmost)
staticAtmost = interp_to_const_cmass_grid(atmost, cmass, cmass[-1])

with open('StaticAtmost.pickle', 'wb') as pkl:
    pickle.dump(staticAtmost, pkl)