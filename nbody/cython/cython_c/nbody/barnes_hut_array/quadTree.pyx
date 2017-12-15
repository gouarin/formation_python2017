# distutils: language = c++
# distutils: sources  = nbody/barnes_hut_array/quadtree.cpp
# cython: boundscheck=False
cimport quadtree
cimport numpy as cnp
import numpy  as  np

cdef class quadArray:
    cdef quadtree.quadArray* c_quadArray# hold an instance of C++ quadArray class
	
    def __cinit__( self, bmin, bmax, size ):
        cdef double b_min[2]
        cdef double b_max[2]
        b_min[0] = bmin[0]
        b_min[1] = bmin[1]
        b_max[0] = bmax[0]
        b_max[1] = bmax[1]
        self.c_quadArray = new quadtree.quadArray( b_min, b_max, size )

    def buildTree( self, double[:,:] particles ):
        self.c_quadArray.build_tree( &particles[0,0] )

    def computeMassDistribution(self, double[:,:] particles, double[:] mass):
        self.c_quadArray.compute_mass_distribution( &particles[0,0], &mass[0] )

    def computeForce(self, double[:] p):
        cdef double acc[2]
        self.c_quadArray.compute_force( &p[0], acc)
        return np.array([acc[0],acc[1]])

    @property
    def ncell(self):
        return self.c_quadArray.ncell()

    @property
    def nbodies(self) :
        return self.c_quadArray.nbodies()

    @property
    def child(self):
        cdef cnp.npy_intp szChild = 4*(2*self.nbodies+1)
        cdef cnp.ndarray[cnp.int, ndim=1] pchild
        pchild = cnp.PyArray_SimpleNewFromData(1,&szChild, cnp.NPY_INT32, 
                                               self.c_quadArray.child())
        return pchild

    @property
    def cell_center(self) :
        cdef cnp.npy_intp szTree = 2*self.nbodies + 1
        cdef cnp.npy_intp dimArr[2]
        cdef cnp.ndarray[cnp.float64_t,ndim=2] ccenter
        dimArr[0] = szTree
        dimArr[1] = 2
        ccenter = cnp.PyArray_SimpleNewFromData(2,&dimArr[0], cnp.NPY_FLOAT64, 
                                                self.c_quadArray.cell_center())
        return ccenter

    @property
    def cell_radius(self) :
        cdef cnp.npy_intp szTree = 2*self.nbodies + 1
        cdef cnp.npy_intp dimArr[2]
        cdef cnp.ndarray[cnp.float64_t  ,ndim=2] cradius
        dimArr[0] = szTree
        dimArr[1] = 2
        cradius = cnp.PyArray_SimpleNewFromData(2,&dimArr[0], cnp.NPY_FLOAT64, 
                                                self.c_quadArray.cell_radius())
        return cradius

    def __str__(self):
        indent = ' '*2
        s = 'Tree :\n'
        for i in range(self.ncell+1):
            s += indent + 'cell {i}\n'.format(i=i)
            cellElements = self.child[self.nbodies + 4*i:self.nbodies + 4*i+4]
            s += 2*indent + 'box: {min} {max} \n'.format(min = self.cell_center[i]-self.cell_radius[i], max = self.cell_center[i]+self.cell_radius[i])
            s += 2*indent + 'particules: {p}\n'.format(p=cellElements[np.logical_and(0<=cellElements, cellElements<self.nbodies)])
            s += 2*indent + 'cells: {c}\n'.format(c=cellElements[cellElements>=self.nbodies]-self.nbodies)
            
        return s