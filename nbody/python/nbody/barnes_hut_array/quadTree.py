import numpy as np
from ..forces import force

class quadArray:
    def __init__(self, bmin, bmax, size):
        self.nbodies = size
        self.child = -np.ones(4*(2*size+1), dtype=np.int32)
        self.bmin = np.asarray(bmin)
        self.bmax = np.asarray(bmax)
        self.center = .5*(self.bmin + self.bmax)
        self.box_size = (self.bmax - self.bmin).max()
        self.ncell = 0
        self.cell_center = np.zeros((2*size+1, 2))
        self.cell_radius = np.zeros(2*size+1)
        self.cell_center[0] = self.center
        self.cell_radius[0] = self.box_size

    def buildTree(self, particles):
        for ip, p in enumerate(particles):
            center = self.center.copy()
            box_size = self.box_size
            x, y = p[:2]
            cell = 0

            childPath = 0
            if x > center[0]:
                childPath += 1
            if y > center[1]:
                childPath += 2

            childIndex = self.nbodies + childPath

            while (self.child[childIndex] > self.nbodies):
                cell = self.child[childIndex] - self.nbodies
                center[:] = self.cell_center[cell]
                childPath = 0
                if x > center[0]:
                    childPath += 1
                if y > center[1]:
                    childPath += 2
                childIndex = self.nbodies + 4*cell + childPath
            # no particle on this cell, just add it
            if (self.child[childIndex] == -1):
                self.child[childIndex] = ip
                self.child[ip] = cell
            # this cell already has a particle
            # subdivide and set the two particles
            elif (self.child[childIndex] < self.nbodies):
                npart = self.child[childIndex]

                oldchildPath = newchildPath = childPath
                while (oldchildPath == newchildPath):
                    self.ncell += 1
                    self.child[childIndex] = self.nbodies + self.ncell 
                    center[:] = self.cell_center[cell]
                    box_size = .5*self.cell_radius[cell]
                    if (oldchildPath&1):
                        center[0] += box_size
                    else:
                        center[0] -= box_size
                    if ((oldchildPath>>1)&1):
                        center[1] += box_size
                    else:
                        center[1] -= box_size

                    oldchildPath = 0
                    if particles[npart][0] > center[0]:
                        oldchildPath += 1
                    if particles[npart][1] > center[1]:
                        oldchildPath += 2

                    newchildPath = 0
                    if p[0] > center[0]:
                        newchildPath += 1
                    if p[1] > center[1]:
                        newchildPath += 2

                    cell = self.ncell

                    self.cell_center[self.ncell] = center
                    self.cell_radius[self.ncell] = box_size

                    childIndex = self.nbodies + 4*self.ncell + oldchildPath

                self.child[childIndex] = npart
                self.child[npart] = self.ncell

                childIndex = self.nbodies + 4*self.ncell + newchildPath
                self.child[childIndex] = ip
                self.child[ip] = self.ncell

    def computeMassDistribution(self, particles, mass):
        self.mass = np.zeros(self.nbodies + self.ncell + 1)
        self.mass[:self.nbodies] = mass
        self.center_of_mass = np.zeros((self.nbodies + self.ncell + 1, 2))
        self.center_of_mass[:self.nbodies] = particles[:, :2]
        for i in range(self.ncell, -1, -1):
            elements = self.child[self.nbodies + 4*i:self.nbodies + 4*i + 4]
            #print('elements', i, elements, self.center_of_mass[elements[elements>=0]]*self.mass[elements[elements>=0]])
            self.mass[self.nbodies + i] = np.sum(self.mass[elements[elements>=0]])
            self.center_of_mass[self.nbodies + i] = np.sum(self.center_of_mass[elements[elements>=0]]*self.mass[elements[elements>=0], np.newaxis], axis=0)
            self.center_of_mass[self.nbodies + i] /= self.mass[self.nbodies + i]
        # print('mass', self.mass)
        # print('center_of_mass', self.center_of_mass)

    def computeForce(self, p):
        depth = 0
        localPos = np.zeros(2*self.nbodies, dtype=np.int32)
        localNode = np.zeros(2*self.nbodies, dtype=np.int32)
        localNode[0] = self.nbodies

        pos = p[:2]
        acc = np.zeros(2)

        while depth >= 0:
            while localPos[depth] < 4:
                child = self.child[localNode[depth] + localPos[depth]]
                # print('child 1', child, localNode[depth] + localPos[depth])
                localPos[depth] += 1
                if child >= 0:
                    if child < self.nbodies:
                        F = force(pos, self.center_of_mass[child], self.mass[child])
                        acc += F
                    else:
                        dx = self.center_of_mass[child, 0] - pos[0]
                        dy = self.center_of_mass[child, 1] - pos[1]
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist != 0 and self.cell_radius[child - self.nbodies]/dist <.5:
                            F = force(pos, self.center_of_mass[child], self.mass[child])
                            acc += F
                        else:
                            depth += 1
                            localNode[depth] = self.nbodies + 4*(child-self.nbodies)
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
