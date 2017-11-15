#!/usr/bin/env python
#-*- coding: utf-8 -*-

from OpenGL.GL   import *
from OpenGL.GLUT import *
import OpenGL.arrays.vbo as glvbo

import sys
import math
import numpy as np
import time
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

    def __init__(self, simu, axis=[0, 1, 0, 1], size=[640, 480], title="Animation", use_colors = False, update_colors = True, start_paused = False):
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
        use_colors: bool
            True to colorize the stars using simu.colors method.
        update_colors: bool
            True if the color must be update at each frame (and not only at the initialisation).
        start_paused: bool
            True if the simulation is initially paused.
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

        # Create a Vertex Buffer Object for the vertices
        coords = simu.coords()
        self.vbo_vertex = glvbo.VBO( coords )
        self.count = coords.shape[0]

        # Displaying colors
        self.update_colors = update_colors
        if use_colors:
            self.toggle_colors()
        else:
            self.use_colors = False

        # Displaying fps
        self.toggle_fps()

        # Paused ?
        self.is_paused = start_paused


    def main_loop(self):
        """ Simulation main loop. """
        glutMainLoop()

    def draw_next_frame(self):
        """ Update simulation data and display it. """
        if not self.is_paused:
            self.simu.next()
            self._update_coords()

            if self.use_colors and self.update_colors:
                self._update_colors()

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
        self.vbo_vertex.set_array(coords)
        self.count = coords.shape[0]

    def _update_colors(self):
        """ Update or create Vertex Buffer Object of colors. """
        colors = self.simu.colors()

        try:
            self.vbo_color.set_array(colors)
        except AttributeError:
            self.vbo_color = glvbo.VBO(colors)

        self.vbo_color.bind()
        if colors.shape[1] == 3:
            glColorPointer(3, GL_DOUBLE, 0, None)
        else:
            glColorPointer(4, GL_DOUBLE, 0, None)

    def toggle_colors(self):
        """ Toggle color display. """
        try:
            self.use_colors = not self.use_colors
        except AttributeError:
            self.use_colors = True

        if self.use_colors:
            try:
                self._update_colors()
                glEnableClientState(GL_COLOR_ARRAY)
            except AttributeError:
                self.use_colors = False

        if not self.use_colors:
            glDisableClientState(GL_COLOR_ARRAY)


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
        elif key == b'f':
            self.toggle_fps()
        elif key == b'c':
            self.toggle_colors()
        elif key == b'p':
            self.is_paused = not self.is_paused

    def _resize(self, width, height):
        """ Called when the window is resized. """
        self.size  = [ max(width, 1), max(height, 1) ]

        # Update the viewport
        glViewport(0, 0, self.size[0], self.size[1])

    def _print(self, text, pos = None, color = [1., 1., 1., 1.]):
        """ Print a text. """

        # Default position is the top-left corner
        if pos is None:
            pos = [ self.axis.origin[0], self.axis.origin[1] + (self.size[1]-15)*self.axis.scale ]

        glColor4f( *color )
        glRasterPos2f( *pos )

        for char in text:
            glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(char))

    def _fps(self):
        """ Update frame time and id and return the fps. """
        curr_time = time.time()
        duration = curr_time - self.frame_times[self.frame_id]
        self.frame_times[self.frame_id] = curr_time
        self.frame_id = (self.frame_id + 1) % len(self.frame_times)

        return len(self.frame_times) / duration

    def toggle_fps(self):
        """ Toggle the display of the fps. """
        try:
            self.use_fps = not self.use_fps
        except AttributeError:
            self.use_fps = True

        if self.use_fps:
            # To calculate the fps
            self.frame_times = [time.time()] * 50
            self.frame_id    = 0


    def _draw(self):
        """ Called when the window must be redrawn. """

        # Alpha blending
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Clear the buffer
        glClear(GL_COLOR_BUFFER_BIT)

        # Update perspective transformation
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(self.axis.origin[0], self.axis.origin[0] + self.axis.scale * self.size[0],
                self.axis.origin[1], self.axis.origin[1] + self.axis.scale * self.size[1],
                -1, 1)

        # Background color
        glClearColor(0., 0., 0., 0.)

        # Draw color
        glColor(1., 1., 1.)

        # Printing fps
        if self.use_fps:
            self._print( "{:.1f}fps".format(self._fps()) )

        # Bind the vertex VBO
        self.vbo_vertex.bind()

        # Tell OpenGL that the VBO contains an array of vertices
        glEnableClientState(GL_VERTEX_ARRAY)

        # These vertices contain 2 double precision coordinates
        glVertexPointer(2, GL_DOUBLE, 0, None)

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

