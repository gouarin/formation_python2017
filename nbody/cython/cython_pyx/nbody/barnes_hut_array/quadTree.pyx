# cython: profile=True
import numpy as np
from ..physics import gamma_si, eps
#cimport forces
#from ..forces cimport force
import cython
cimport libc.math as cmath

cdef double g_si, epsi
g_si = gamma_si
epsi = eps

@cython.boundscheck(False)
@cython.cdivision(True)
cdef void force( const double* p1, const double* p2, double m2, double* ret ) :
    cdef :
        double dx, dy
        double dist
        double F
    dx   = p2[0] - p1[0]
    dy   = p2[1] - p1[1]
    dist = cmath.sqrt( dx**2 + dy**2 + epsi )

    F = 0.
    if  dist > 0 :
        F  = ( g_si * m2 ) / ( dist * dist * dist )
    ret[0] = F * dx
    ret[1] = F * dy


class quadArray:
    def __init__(self, bmin, bmax, size):
        self.nbodies = size
        self.child = -np.ones(4*(2*size+1), dtype=np.int32)
        self.bmin = np.asarray(bmin)
        self.bmax = np.asarray(bmax)
        self.center = .5*(self.bmin + self.bmax)
        self.box_size = (self.bmax - self.bmin)
        self.ncell = 0
        self.cell_center = np.zeros((2*size+1, 2))
        self.cell_radius = np.zeros((2*size+1, 2))
        self.cell_center[0] = self.center
        self.cell_radius[0] = self.box_size

    @cython.boundscheck(False)
    def buildTree(self, double[:,:] particles):
        cdef:
            int ip, cell, childPath, nbodies, childIndex, ncell
            int newchildPath, oldchildPath, npart
            double c_center[2]
            double[:] center
            double c_box_size[2]
            double[:] box_size
            double[:] center_view
            double[:] box_size_view
            double[:,:] cell_center_view
            double[:,:] cell_radius_view
            int[:] child_view
            double x,y
        center_view = self.center
        box_size_view = self.box_size
        center = c_center
        box_size = c_box_size
        child_view = self.child
        cell_radius_view = self.cell_radius
        cell_center_view = self.cell_center

        ncell = 0
        nbodies = self.nbodies
        for ip in range(nbodies):
#        for ip, p in enumerate(particles):
#            center = self.center.copy()
            center[0] = center_view[0]
            center[1] = center_view[1]
            box_size[0] = box_size_view[0]
            box_size[1] = box_size_view[1]
            x = particles[ip,0]
            y = particles[ip,1]
            #x, y = p[:2]
            cell = 0

            childPath = 0
            if x > center[0]:
                childPath += 1
            if y > center[1]:
                childPath += 2

            childIndex = nbodies + childPath

            while (child_view[childIndex] > nbodies):
                cell = child_view[childIndex] - nbodies
                center[0] = cell_center_view[cell,0]
                center[1] = cell_center_view[cell,1]
                childPath = 0
                if x > center[0]:
                    childPath += 1
                if y > center[1]:
                    childPath += 2
                childIndex = nbodies + 4*cell + childPath
            # no particle on this cell, just add it
            if (child_view[childIndex] == -1):
                child_view[childIndex] = ip
                child_view[ip] = cell
            # this cell already has a particle
            # subdivide and set the two particles
            elif (child_view[childIndex] < nbodies):
                npart = child_view[childIndex]

                oldchildPath = newchildPath = childPath
                while (oldchildPath == newchildPath):
                    ncell += 1
                    child_view[childIndex] = nbodies + ncell 
                    center[0] = cell_center_view[cell,0]
                    center[1] = cell_center_view[cell,1]
                    box_size[0] = .5*cell_radius_view[cell,0]
                    box_size[1] = .5*cell_radius_view[cell,1]
                    if (oldchildPath&1):
                        center[0] += box_size[0]
                    else:
                        center[0] -= box_size[0]
                    if ((oldchildPath>>1)&1):
                        center[1] += box_size[1]
                    else:
                        center[1] -= box_size[1]

                    oldchildPath = 0
                    if particles[npart,0] > center[0]:
                        oldchildPath += 1
                    if particles[npart,1] > center[1]:
                        oldchildPath += 2

                    newchildPath = 0
                    if particles[ip,0] > center[0]:
                        newchildPath += 1
                    if particles[ip,1] > center[1]:
                        newchildPath += 2

                    cell = ncell

                    cell_center_view[ncell,0] = center[0]
                    cell_center_view[ncell,1] = center[1]
                    cell_radius_view[ncell,0] = box_size[0]
                    cell_radius_view[ncell,1] = box_size[1]

                    childIndex = nbodies + 4*ncell + oldchildPath

                child_view[childIndex] = npart
                child_view[npart] = ncell

                childIndex = nbodies + 4*ncell + newchildPath
                child_view[childIndex] = ip
                child_view[ip] = ncell
        self.ncell = ncell

    @cython.boundscheck(False)
    @cython.cdivision(True)
    def computeMassDistribution(self, double[:,:] particles, double[:] mass):
        cdef:
            int i, j, ncell, nbodies, ej
            double[:,:] particles_view
            double[:]   smass_view
            double[:,:] scenter_of_mass
            int elements[4]
            int[:] child_view

        ncell = self.ncell
        nbodies = self.nbodies
        child_view      = self.child

        self.mass = np.zeros(nbodies + ncell + 1)
        smass_view = self.mass
        smass_view[:nbodies] = mass
        #self.mass[:self.nbodies] = mass
        self.center_of_mass = np.zeros((nbodies + ncell + 1, 2))
        scenter_of_mass = self.center_of_mass
        #self.center_of_mass[:self.nbodies] = particles[:, :2]
        scenter_of_mass[:nbodies] = particles[:, :2]
        for i in range(ncell, -1, -1):
            elements[0] = child_view[nbodies + 4*i    ]
            elements[1] = child_view[nbodies + 4*i + 1]
            elements[2] = child_view[nbodies + 4*i + 2]
            elements[3] = child_view[nbodies + 4*i + 3]
            #print('elements', i, elements, self.center_of_mass[elements[elements>=0]]*self.mass[elements[elements>=0]])
            smass_view[nbodies + i] = 0.
            scenter_of_mass[nbodies + i,0] = 0.
            scenter_of_mass[nbodies + i,1] = 0.
            for j in range(4):
                ej = elements[j]
                if (ej >= 0):
                    smass_view[nbodies + i] += smass_view[ej]
                    scenter_of_mass[nbodies + i,0] += scenter_of_mass[ej,0]*smass_view[ej]
                    scenter_of_mass[nbodies + i,1] += scenter_of_mass[ej,1]*smass_view[ej]
            #scenter_of_mass[nbodies + i] = np.sum(self.center_of_mass[elements[elements>=0]]*self.mass[elements[elements>=0], np.newaxis], axis=0)
            scenter_of_mass[nbodies + i,0] /= smass_view[nbodies + i]
            scenter_of_mass[nbodies + i,1] /= smass_view[nbodies + i]
        #print('mass', self.mass)
        #print('center_of_mass', self.center_of_mass)

    @cython.boundscheck(False)
    @cython.cdivision(True)
    def computeForce(self, double[:] p):
        cdef:
            int depth, child
            int[:] localPos
            int[:] localNode
            double dx
            double dy
            double dist
            double acc[2]
            double F[2]
            double[:,:] center_of_mass_view
            double[:] mass_view
            double[:,:] radius_view
            int[:] child_view
            int nbodies

        center_of_mass_view = self.center_of_mass
        mass_view           = self.mass
        radius_view         = self.cell_radius
        child_view          = self.child
        nbodies = self.nbodies

        depth = 0
        localPos = np.zeros(2*nbodies, dtype=np.int32)
        localNode = np.zeros(2*nbodies, dtype=np.int32)
        localNode[0] = nbodies
        acc[0] = 0
        acc[1] = 0
        while depth >= 0:
            while localPos[depth] < 4:
                child = child_view[localNode[depth] + localPos[depth]]
                # print('child 1', child, localNode[depth] + localPos[depth])
                localPos[depth] += 1
                if child >= 0:
                    if child < nbodies:
                        #F = force(pos, center_of_mass_view[child], mass_view[child])
                        force(&p[0], &center_of_mass_view[child,0], mass_view[child], &F[0])
                        acc[0] += F[0]
                        acc[1] += F[1]
                    else:
                        dx = center_of_mass_view[child, 0] - p[0]
                        dy = center_of_mass_view[child, 1] - p[1]
                        dist = cmath.sqrt(dx**2 + dy**2)
                        if dist != 0 and radius_view[child - nbodies,0]/dist <.5:
                            #F = force(pos, center_of_mass_view[child], mass_view[child])
                            force(&p[0], &center_of_mass_view[child,0], mass_view[child], &F[0])
                            acc[0] += F[0]
                            acc[1] += F[1]
                        else:
                            depth += 1
                            localNode[depth] = nbodies + 4*(child-nbodies)
                            localPos[depth] = 0
            depth -= 1
        return acc


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