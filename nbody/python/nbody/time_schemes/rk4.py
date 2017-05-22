import numpy as np

class RK4:
    def __init__(self, dt, nbodies, method):
        self.dt = dt
        self.method = method
        self.k1 = np.zeros((nbodies, 4)) 
        self.k2 = np.zeros((nbodies, 4)) 
        self.k3 = np.zeros((nbodies, 4)) 
        self.k4 = np.zeros((nbodies, 4)) 
        self.tmp = np.zeros((nbodies, 4))

    def init(self, mass, particles):
        pass

    def update(self, mass, particles):
        # k1
        self.method(mass, particles, self.k1)
        self.tmp[:, :] = particles[:, :4] + self.dt*0.5*self.k1

        # k2
        self.method(mass, self.tmp, self.k2)
        self.tmp[:, :] = particles[:, :4] + self.dt*0.5*self.k2

        # k3
        self.method(mass, self.tmp, self.k3)
        self.tmp[:, :] = particles[:, :4] + self.dt*self.k3

        # k4
        self.method(mass, self.tmp, self.k4)

        particles[:, :] += self.dt/6*(self.k1 + 2*(self.k2+self.k3) + self.k4)