#!/usr/bin/env python
#-*- coding: utf-8 -*-

from OpenGL.GL   import *
from OpenGL.GLUT import *
import OpenGL.arrays.vbo as glvbo

import sys
import math
import numpy as np
from copy import deepcopy


class Axis:
    """ View axis. """
    def __init__(self, origin, scale):
        self.origin = origin
        self.scale  = scale

class Animation:
    """ Simulation renderer using OpenGL.

    Press left button to move.
    Press right button to zoom.

    Freely inspired from:
        http://cyrille.rossant.net/2d-graphics-rendering-tutorial-with-pyopengl/
        https://sites.google.com/site/dlampetest/python/vectorized-particle-system-and-geometry-shaders
        http://carloluchessa.blogspot.fr/2012/09/simple-viewer-in-pyopengl.html
    """

    def __init__(self, simu, axis=[0, 1, 0, 1], size=[640, 480], title="Animation"):
        """ Initialize an animation view.

        Parameters:
        -----------
        simu: object
            Simulation object with coords and next methods
        axis: list
            Axis bounds [ xmin, xmax, ymin, ymax ].
        size: list
            Initial window size [width, height].
        title: string
            Window title.
        """

        self.simu   = simu
        self.axis   = Axis( [ axis[0], axis[2] ], max((axis[1]-axis[0])/size[0], (axis[3]-axis[2])/size[1]) )
        self.size   = size
        self.action = None

        # Initialize the OpenGL Utility Toolkit
        glutInit(sys.argv)

        # Initial display mode (RGBA colors and double buffered window)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE)

        # Initial window
        glutInitWindowSize(size[0], size[1])
        glutInitWindowPosition(0, 0)
        glutCreateWindow(title)

        # Callbacks
        glutDisplayFunc(self._draw)         # When the window must be redrawn
        glutIdleFunc(self.draw_next_frame)  # When OpenGL gets bored
        glutReshapeFunc(self._resize)       # When the window is resized
        glutMouseFunc(self._mouse)          # When a mouse button is pressed/released
        glutMotionFunc(self._motion)        # When the mouse move with a pressed button
        glutKeyboardFunc(self._keyboard)    # When a key is pressed

        # Background color
        glClearColor(0., 0., 0., 0.)

        # Create a Vertex Buffer Object
        coords = simu.coords()
        self.vbo   = glvbo.VBO( coords )
        self.count = coords.shape[0]


    def main_loop(self):
        """ Simulation main loop. """
        glutMainLoop()

    def draw_next_frame(self):
        """ Update simulation data and display it. """
        self.simu.next()
        self._update_coords()
        glutPostRedisplay()

    def reset_view(self):
        """ Reset view accordingly to the particle coordinates. """
        border = 0.1
        coords = self.simu.coords()
        coord_min = coords.min(axis=0)
        coord_max = coords.max(axis=0)

        self.axis.scale = (1. + 2*border) * max( (coord_max[0]-coord_min[0])/self.size[0], (coord_max[1]-coord_min[1])/self.size[1] )
        self.axis.origin = [
            0.5*(coord_min[0] + coord_max[0] - self.size[0]*self.axis.scale),
            0.5*(coord_min[1] + coord_max[1] - self.size[1]*self.axis.scale)
        ]

        glutPostRedisplay()

    def _update_coords(self):
        """ Update vertex coordinates. """
        coords = self.simu.coords()
        self.vbo.set_array(coords)
        self.count = coords.shape[0]

    def _mouse(self, button, state, x, y):
        """ Called when a mouse button has been pressed/released. """
        if self.action is None and state == GLUT_DOWN:
            self.button = button
            self.old_axis = deepcopy(self.axis)
            self.x_start = x
            self.y_start = y

            if button == GLUT_LEFT_BUTTON:
                self.action = 'move'
            elif button == GLUT_RIGHT_BUTTON:
                self.action = 'zoom'

        elif self.action is not None and state == GLUT_UP and button == self.button:
            self.action = None


    def _motion(self, x, y):
        """ Called when the mouse has move while a button is pressed. """
        if self.action == 'move':
            self.axis.origin[0] = self.old_axis.origin[0] - self.old_axis.scale * (x - self.x_start)
            self.axis.origin[1] = self.old_axis.origin[1] + self.old_axis.scale * (y - self.y_start)

        elif self.action == 'zoom':
            zoom_factor = math.exp( 0.01 * (self.y_start - y) )
            self.axis.origin[0] = self.old_axis.origin[0] + (1 - zoom_factor) * self.old_axis.scale * self.x_start
            self.axis.origin[1] = self.old_axis.origin[1] + (1 - zoom_factor) * self.old_axis.scale * (self.size[1]-self.y_start)
            self.axis.scale = zoom_factor * self.old_axis.scale

        glutPostRedisplay()

    def _keyboard(self, key, x, y):
        """ Called when a key is pressed. """
        if   key == b'r':
            self.reset_view()
        elif key == b'q':
            glutLeaveMainLoop()

    def _resize(self, width, height):
        """ Called when the window is resized. """
        self.size  = [ max(width, 1), max(height, 1) ]

        # Update the viewport
        glViewport(0, 0, self.size[0], self.size[1])


    def _draw(self):
        """ Called when the window must be redrawn. """

        # Clear the buffer
        glClear(GL_COLOR_BUFFER_BIT)

        # Update perspective transformation
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(self.axis.origin[0], self.axis.origin[0] + self.axis.scale * self.size[0],
                self.axis.origin[1], self.axis.origin[1] + self.axis.scale * self.size[1],
                -1, 1)

        # Draw color
        glColor(1., 1., 1.)

        # Bind the VBO
        self.vbo.bind()

        # Tell OpenGL that the VBO contains an array of vertices
        glEnableClientState(GL_VERTEX_ARRAY)

        # These vertices contain 2 double precision coordinates
        glVertexPointer(2, GL_DOUBLE, 0, self.vbo)

        # Draw "count" points from the VBO
        glDrawArrays(GL_POINTS, 0, self.count)

        # Swap display buffers
        glutSwapBuffers()



if __name__ == '__main__':
    """ Demo """
    import numpy as np

    class SpinningCloud:
        def __init__(self, size, theta = math.pi/18000):
            self._coords = np.array(np.random.randn(size, 2), dtype = np.float64)
            self._rot    = np.asarray([[ math.cos(theta), math.sin(theta) ],[-math.sin(theta), math.cos(theta) ]])

        def next(self):
            self._coords = np.dot( self._coords, self._rot )

        def coords(self):
            return self._coords

    simu = SpinningCloud(100000, math.pi/1800)

    anim = Animation( simu, axis=[-1, 1, -1, 1] )
    anim.main_loop()

