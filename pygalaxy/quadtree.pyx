""" A Cython implementation of a QuadTree and other useful classes"""
import numpy as np
cimport numpy as np

# Type of the numpy arrays in Python namespace
DTYPE = np.double
# Correspondant C type
ctypedef np.double_t DTYPE_T

cdef class Body:
    """ A body to be inserted in the three """

    # Position
    cdef public np.ndarray pos

    # Velocity
    cdef public np.ndarray vel

    # Mass
    cdef public double mass

    # Force acting on this body
    cdef public np.ndarray force

    def __cinit__(self, np.ndarray pos, np.ndarray vel, mass):
        self.pos = pos
        self.vel = vel
        self.mass = mass
