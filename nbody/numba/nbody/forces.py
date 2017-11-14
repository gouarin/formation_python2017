from math import sqrt
from .physics import gamma_si, eps, gamma_1
import numpy as np
import numba

@numba.njit
def force(p1, p2, m2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist = sqrt(dx**2 + dy**2 + eps)

    F = 0.
    if dist > 0:
        F = (gamma_si * m2) / (dist*dist*dist)

    return F * dx, F * dy
