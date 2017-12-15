# cython: profile=True
from libc.math cimport sqrt
from .physics import gamma_si, eps
cimport cython

cdef double g_si, epsi
g_si = gamma_si
epsi = eps


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def force( cython.floating[:] p1, cython.floating[:] p2, cython.floating m2):
    cdef:
        cython.floating dx
        cython.floating dy
        cython.floating dist
        cython.floating F
        cython.floating ret[2]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist = sqrt(dx**2 + dy**2 + epsi)

    F = 0.
    if dist > 0:
        F = (g_si * m2) / (dist*dist*dist)
    ret[0] = F * dx
    ret[1] = F * dy
    return ret
    #return [F * dx, F * dy]