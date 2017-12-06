#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Animation(object):
    """ Simulation renderer using matplotlib. """

    def __init__(self, simu, axis=[0, 1, 0, 1]):
        """ Initialize an animation view.

        Parameters:
        -----------
        simu: object
            Simulation object with coords and next methods
        axis: list
            Axis bounds [ xmin, xmax, ymin, ymax ].
        """

        self.simu = simu

        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('black')
        self.ax.axis(axis)

        coords = simu.coords()
        self.scatter = plt.scatter(coords[:, 0], coords[:, 1], c='white', s=.5)

    def _update_coords(self, i):
        """ Update scatter coordinates. """
        self.simu.next()
        self.scatter.set_offsets(self.simu.coords())

        # We need to return an iterable since FuncAnimation expects a returned
        # object of this nature
        return self.scatter,

    def main_loop(self):
        """ Animation main loop. """
        # We need to keep the animation object around otherwise it is garbage
        # collected. So we use the dummy `_`
        _ = animation.FuncAnimation(self.fig, self._update_coords, blit=True)
        plt.show()
