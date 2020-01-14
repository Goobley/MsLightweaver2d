import numpy as np
from dataclasses import dataclass

@dataclass
class FangH:
    C1c: np.ndarray
    C12: np.ndarray
    C13: np.ndarray
    C14: np.ndarray


def fang_ele_rates_H(neutralH, ne1, bheat1):
    clog = 24.68
    clog1 = 8.13
    gam = ne1 * clog + neutralH * clog1
    coeff = np.maximum(bheat1, 0.0) * clog1 / gam

    C1c = 1.73e10 * coeff
    C12 = 2.94e10 * coeff
    C12 = 5.35e9 * coeff
    C14 = 1.91e9 * coeff

    return FangH(C1c, C12, C13, C14)

