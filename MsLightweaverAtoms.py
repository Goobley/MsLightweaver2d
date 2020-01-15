from lightweaver.rh_atoms import H_6_atom, He_9_atom, CaII_atom
from lightweaver.atomic_model import reconfigure_atom
from Fang import FangHRates

def H_6():
    H = H_6_atom()
    # for l in H.lines:
    #     l.NlambdaGen *= 2
    H.collisions.append(FangHRates(0,0))

    reconfigure_atom(H)
    return H

def CaII():
    Ca = CaII_atom()
    # for l in Ca.lines:
    #     l.NlambdaGen *= 2

    # reconfigure_atom(Ca)
    return Ca

def He_9():
    He = He_9_atom()
    # for l in He.lines:
    #     l.NlambdaGen *= 2

    # reconfigure_atom(He)
    return He