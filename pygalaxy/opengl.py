#!/usr/bin/env python

from OpenGL.GL import *
from OpenGL.GLUT import *
import OpenGL.arrays.vbo as glvbo
from OpenGL.GL import shaders

import sys
import math
import numpy as np
import time
from copy import deepcopy


class Animation(object):
    """ Simulation renderer using OpenGL.

    Press left button to move.
    Press right button to zoom.
    Other shortcuts to control display: 'h' to se the help.


    Under *Windows*, in order to have a working GLUT install,
    follow these instructions:
        https://codeyarns.com/2012/04/27/pyopengl-installation-notes-for-windows/
        https://deparkes.co.uk/2015/02/04/anaconda-whl-install/
    1) uninstall pyopengl and pyopengl-accelerate Python modules,
    2) download appropriate files (windows and python versions) from
        https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopengl,
    3) install it using 'pip install <file>' (even if you use Anaconda).


    Freely inspired from:
        http://cyrille.rossant.net/2d-graphics-rendering-tutorial-with-pyopengl/
        https://sites.google.com/site/dlampetest/python/vectorized-particle-system-and-geometry-shaders
        http://carloluchessa.blogspot.fr/2012/09/simple-viewer-in-pyopengl.html
    """

    def __init__(self, simu,
                 axis=[0., 1., 0., 1.], size=[640, 480], title=b"Animation",
                 use_colors=False, update_colors=False,
                 use_adaptative_opacity=False, start_paused=False):
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
            True if the color must be update at each frame (and not only at
                the initialisation).
        use_adaptative_opacity: bool
            True if the opacity is adapted to the view zoom.
        start_paused: bool
            True if the simulation is initially paused.
        """

        self.simu = simu
        faxis = [float(v) for v in axis]
        self.axis = Animation._Axis([faxis[0], faxis[2]],
                                    max((faxis[1]-faxis[0])/size[0],
                                        (faxis[3]-faxis[2])/size[1]))
        self.size = size
        self.mouse_action = None

        # Initialize the OpenGL Utility Toolkit
        glutInit(sys.argv)

        # Initial display mode (RGBA colors and double buffered window)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE)

        # Initial window
        glutInitWindowSize(size[0], size[1])
        glutInitWindowPosition(0, 0)
        glutCreateWindow(title)

        # Callbacks
        # When the window must be redrawn
        glutDisplayFunc(self._draw)

        # When OpenGL gets bored
        glutIdleFunc(self.draw_next_frame)

        # When the window is resized
        glutReshapeFunc(self._resize)

        # When a mouse button is
        # pressed/released
        glutMouseFunc(self._mouse)

        # When the mouse moves with a pressed button
        glutMotionFunc(self._motion)

        # When a key is pressed
        glutKeyboardFunc(self._keyboard)

        # Create a Vertex Buffer Object for the vertices
        coords = simu.coords()
        self._star_vbo = glvbo.VBO(coords)
        self._star_count = coords.shape[0]

        # Display options
        self.use_colors_update = update_colors
        self.use_colors = use_colors
        self.adaptative_opacity_factor = self.axis.scale
        self.use_fps = True
        self.is_paused = start_paused

    ###########################################################################
    # Properties

    @property
    def use_colors(self):
        """ Control color display. """
        try:
            return self.__use_colors
        except AttributeError:
            return False

    @use_colors.setter
    def use_colors(self, value):
        self.__use_colors = value

        if self.__use_colors:
            try:
                if self.use_colors_update or not hasattr(self, 'vbo_color'):
                    self._update_colors()
                glEnableClientState(GL_COLOR_ARRAY)
            except AttributeError as e:
                self.__use_colors = False
                print('Cannot update star colors: {}'.format(str(e)))

        if not self.__use_colors:
            glDisableClientState(GL_COLOR_ARRAY)

    @property
    def use_colors_update(self):
        """ Control color update for each frame. """
        try:
            return self.__use_colors_update
        except AttributeError:
            return False

    @use_colors_update.setter
    def use_colors_update(self, value):
        self.__use_colors_update = value

    @property
    def use_fps(self):
        """ Control the display of the fps. """
        try:
            return self.__use_fps
        except AttributeError:
            return False

    @use_fps.setter
    def use_fps(self, value):
        self.__use_fps = value

        if self.__use_fps:
            # To calculate the fps
            self.frame_times = [time.time()] * 50
            self.frame_id = 0

    @property
    def is_paused(self):
        """ Control the execution of the simulation. """
        try:
            return self.__is_paused
        except AttributeError:
            return False

    @is_paused.setter
    def is_paused(self, value):
        self.__is_paused = value

    @property
    def use_help(self):
        """ Control help display. """
        try:
            return self.__use_help
        except AttributeError:
            return False

    @use_help.setter
    def use_help(self, value):
        self.__use_help = value

    @property
    def use_adaptative_opacity(self):
        """ Control the adaptive opacity.

        Opacity varies linearly with the zoom factor.
        """
        try:
            return self.__use_adaptative_opacity
        except AttributeError:
            return False

    @use_adaptative_opacity.setter
    def use_adaptative_opacity(self, value):
        self.__use_adaptative_opacity = value

        if value and not hasattr(self, '_ao_shader_program'):
            try:
                vertex_shader = shaders.compileShader("""
                    #version 120
                    uniform float opacity_factor;
                    void main()
                    {
                        gl_Position = ftransform();
                        gl_FrontColor = gl_Color;
                        gl_FrontColor[3] = min(1., gl_FrontColor[3]*opacity_factor);
                    }
                    """, GL_VERTEX_SHADER)

                fragment_shader = shaders.compileShader("""
                    #version 120
                    void main()
                    {
                        gl_FragColor = gl_Color;
                    }
                    """, GL_FRAGMENT_SHADER)

                self._ao_shader_program = shaders.compileProgram(
                    vertex_shader,
                    fragment_shader
                )
            except RuntimeError as e:
                self.__use_adaptative_opacity = False
                print('Adaptative opacity failure: {}'.format(str(e)))

    @property
    def adaptative_opacity_factor(self):
        """ Control the adaptative opacity amplitude.

        For a given view/zoom, setting this factor to
        the ratio between the axis width/height and
        the screen width/height (i.e. the zoom factor)
        doesn't change the display.
        """
        try:
            return self.__ao_factor
        except AttributeError:
            self.ao_factor = self.axis.scale
            return self.__ao_factor

    @adaptative_opacity_factor.setter
    def adaptative_opacity_factor(self, value):
        self.__ao_factor = value

    @property
    def tracked_star(self):
        """ Id of the tracked star. None if disabled. """
        try:
            return self.__tracket_star
        except AttributeError:
            return None

    @tracked_star.setter
    def tracked_star(self, value):
        self.__tracket_star = value

    @property
    def star_radius(self):
        """ Radius of the stars when using nebulae display. """
        try:
            return self.__star_radius
        except AttributeError:
            return 1e-4

    @star_radius.setter
    def star_radius(self, value):
        self.__star_radius = value

    @property
    def star_min_pixel_size(self):
        """ Minimal pixel size of the stars when using nebulae display.

        A value of one correspond to the size of a pixel.
        """
        try:
            return self.__star_min_pixel_size
        except AttributeError:
            return 1.

    @star_min_pixel_size.setter
    def star_min_pixel_size(self, value):
        self.__star_min_pixel_size = value

    @property
    def nebulae_max_radius(self):
        """ Maximal radius of the nebuale. """
        try:
            return self.__nebulae_max_radius
        except AttributeError:
            return 0.3

    @nebulae_max_radius.setter
    def nebulae_max_radius(self, value):
        self.__nebulae_max_radius = value

    @property
    def nebulae_radius_factor(self):
        """ Factor controlling how the nebulae radius varies with the view scale.

        The nebulae decreases to the star radius when zooming in.
        This factor control how fast it converges.

        For a given view/zoom, setting this factor to
        the ratio between the axis width/height and
        the screen width/height (i.e. the zoom factor)
        results in the maximal nebuale rendered radius.
        """
        try:
            return self.__nebulae_radius_factor
        except AttributeError:
            self.__nebulae_radius_factor = self.axis.scale
            return self.__nebulae_radius_factor

    @nebulae_radius_factor.setter
    def nebulae_radius_factor(self, value):
        self.__nebulae_radius_factor = value

    @property
    def nebulae_density_factor(self):
        """ Density reduction factor of the nebulae.

        The opacity of the nubulae varies as
            alpha/r
        with alpha the density redution factor
        and r the distance to the star.
        """
        try:
            return self.__nebulae_density_factor
        except AttributeError:
            return 9e-3

    @nebulae_density_factor.setter
    def nebulae_density_factor(self, value):
        self.__nebulae_density_factor = value

    @property
    def use_pixel_render(self):
        """ Control star rendering as pixels. """
        try:
            return self.__use_pixel_render
        except AttributeError:
            return True

    @use_pixel_render.setter
    def use_pixel_render(self, value):
        self.__use_pixel_render = value

    @property
    def use_nebulae_render(self):
        """ Control star rendering as nebulae. """
        try:
            return self.__use_nebulae_render
        except AttributeError:
            return False

    @use_nebulae_render.setter
    def use_nebulae_render(self, value):
        self.__use_nebulae_render = value

        if value and not hasattr(self, '_nebulae_shader_program'):
            try:
                vertex_shader = shaders.compileShader("""
                    #version 120
                    uniform float opacity_factor;
                    uniform float view_scale;
                    uniform float radius_factor;
                    uniform float max_nebulae_radius;
                    uniform float star_radius;
                    uniform float min_pixel_size;
                    uniform float density_factor;

                    varying float distance_shift;
                    varying float opacity_shift;

                    void main()
                    {
                        gl_Position = ftransform();

                        // Defining the nebulae constraints
                        float max_sprite_radius = max(0.5f * min_pixel_size, max_nebulae_radius/view_scale);
                        float min_sprite_radius = max(0.5f * min_pixel_size, star_radius/view_scale);
                        float nebulae_radius    = min(max_sprite_radius,
                                                      min_sprite_radius + (max_sprite_radius - min_sprite_radius) * view_scale/radius_factor);
                        float max_opacity = min(1.f, gl_Color[3] * opacity_factor);

                        // Finding parameter sets that fits the nebulae constraints
                        float delta    = max_opacity * (max_opacity - 8*density_factor/(min_sprite_radius/nebulae_radius - 1.f));
                        opacity_shift  = 0.5f * ( max_opacity - sqrt(delta) );
                        distance_shift = -density_factor/opacity_shift - 0.5f;

                        // Customizing sprite
                        gl_PointSize = 2.f * nebulae_radius;
                        gl_FrontColor = gl_Color;
                        gl_FrontColor[3] = max_opacity;
                    }
                    """, GL_VERTEX_SHADER)

                fragment_shader = shaders.compileShader("""
                    #version 120
                    uniform float density_factor;

                    varying float distance_shift;
                    varying float opacity_shift;

                    void main()
                    {
                        gl_FragColor = gl_Color;
                        float dist = length(gl_PointCoord - vec2(0.5, 0.5));
                        gl_FragColor[3] = dist > -distance_shift ? clamp(density_factor/(dist + distance_shift) + opacity_shift, 0.f, gl_FragColor[3]) : gl_FragColor[3];
                    }
                    """, GL_FRAGMENT_SHADER)

                self._nebulae_shader_program = shaders.compileProgram(
                    vertex_shader,
                    fragment_shader
                )
            except RuntimeError as e:
                self.__use_nebulae_render = False
                print('Nebulae render failure: {}'.format(str(e)))

    ###########################################################################
    # Public methods

    def main_loop(self):
        """ Simulation main loop. """
        glutMainLoop()

    def draw_next_frame(self):
        """ Update simulation data and display it. """
        if not self.is_paused:
            self.simu.next()
            self._update_coords()

            if self.use_colors and self.use_colors_update:
                self._update_colors()

        glutPostRedisplay()

    def reset_view(self):
        """ Reset view accordingly to the particle coordinates and the
        tracked star. """
        border = 0.1
        coords = self.simu.coords()
        coord_min = coords.min(axis=0)
        coord_max = coords.max(axis=0)

        if self.tracked_star is None:
            center = 0.5 * (coord_min + coord_max)
        else:
            center = coords[self.tracked_star]

        axis_size = 2 * np.maximum(center - coord_min, coord_max - center)
        self.axis.scale = (1. + 2*border) * np.max(axis_size / self.size)
        self.center_view(*center)

    def center_view(self, x, y):
        """ Center view on the given axis coordinates. """
        self.axis.origin = [
            x - 0.5 * self.size[0]*self.axis.scale,
            y - 0.5 * self.size[1]*self.axis.scale
        ]

    def reset_display_params(self):
        """ Set some display parameters accordingly to the current view.

        Concerned parameters are:
        - adaptative_opacity_factor = 1/zoom_factor
        """
        self.adaptative_opacity_factor = self.axis.scale
        self.nebulae_radius_factor = self.axis.scale

    ###########################################################################
    # Internal classes

    class _Axis(object):
        """ View axis. """
        def __init__(self, origin, scale):
            self.origin = origin
            self.scale = scale

    ###########################################################################
    # Internal methods

    def _update_coords(self):
        """ Update vertex coordinates. """
        coords = self.simu.coords()
        self._star_vbo.set_array(coords)
        self._star_count = coords.shape[0]

        # Centering view on tracked star
        if self.tracked_star is not None:
            self.center_view(*coords[self.tracked_star])

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

    def _mouse(self, button, state, x, y):
        """ Called when a mouse button has been pressed/released. """
        if self.mouse_action is None and state == GLUT_DOWN:
            self.button = button
            self.old_axis = deepcopy(self.axis)
            self.x_start = x
            self.y_start = y

            if button == GLUT_LEFT_BUTTON:
                self.mouse_action = 'move'
            elif button == GLUT_RIGHT_BUTTON:
                self.mouse_action = 'zoom'

        elif self.mouse_action is not None \
                and state == GLUT_UP \
                and button == self.button:

            self.mouse_action = None

    def _motion(self, x, y):
        """ Called when the mouse has move while a button is pressed. """
        # Disable view translation when tracking a star,
        # unless simulation is paused.
        if self.mouse_action == 'move' and \
                (self.tracked_star is None or self.is_paused):
            self.axis.origin[0] = self.old_axis.origin[0] - \
                self.old_axis.scale * (x - self.x_start)
            self.axis.origin[1] = self.old_axis.origin[1] + \
                self.old_axis.scale * (y - self.y_start)

        elif self.mouse_action == 'zoom':
            zoom_factor = math.exp(0.01 * (self.y_start - y))

            # By default, zooming is focused on the coordinates initialy
            # pointed by the mouse
            # but if a star is tracked and the simulation is not paused,
            # then the focus is on this star.
            if self.tracked_star is None or self.is_paused:
                self.axis.origin[0] = self.old_axis.origin[0] + \
                    (1 - zoom_factor) * self.old_axis.scale * self.x_start

                self.axis.origin[1] = self.old_axis.origin[1] + \
                    (1 - zoom_factor) * self.old_axis.scale * \
                    (self.size[1]-self.y_start)

                self.axis.scale = zoom_factor * self.old_axis.scale
            else:
                star_coords = self.simu.coords()[self.tracked_star]
                self.axis.scale = zoom_factor * self.old_axis.scale
                self.center_view(star_coords[0], star_coords[1])

        glutPostRedisplay()

    def _keyboard(self, key, x, y):
        """ Called when a key is pressed. """
        if key == b'r':
            self.reset_view()
        elif key == b'R':
            self.reset_display_params()
        elif key == b'q':
            glutLeaveMainLoop()
        elif key == b'f':
            self.use_fps = not self.use_fps
        elif key == b'c':
            self.use_colors = not self.use_colors
        elif key == b'u':
            self.use_colors_update = not self.use_colors_update
        elif key == b'o':
            self.use_adaptative_opacity = not self.use_adaptative_opacity
        elif key == b's':
            self.use_pixel_render = not self.use_pixel_render
        elif key == b'n':
            self.use_nebulae_render = not self.use_nebulae_render
        elif key == b't':
            self.tracked_star = self._find_nearest_star(x, y)
        elif key == b'T':
            self.tracked_star = None
        elif key == b'p' or key == b' ':
            self.is_paused = not self.is_paused
        elif key == b'h':
            self.use_help = not self.use_help

    def _resize(self, width, height):
        """ Called when the window is resized. """
        self.size = [max(width, 1), max(height, 1)]

        # Update the viewport
        glViewport(0, 0, self.size[0], self.size[1])

    def _print(self, text, pos=[0, 0], color=[1., 1., 1., 1.]):
        """ Print a text. """

        # Default position is the top-left corner
        pos = [self.axis.origin[0] + 9*pos[0]*self.axis.scale,
               self.axis.origin[1] + \
               (self.size[1] - 15*(pos[1]+1))*self.axis.scale]

        glColor4f(*color)
        glRasterPos2f(*pos)

        for char in text:
            if char == "\n":
                pos[1] -= 15*self.axis.scale
                glRasterPos2f(*pos)
            else:
                glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(char))

    def _print_help(self):
        """ Print help. """
        self._print(pos=[0, 1], text="""left click: translate view
right click: zoom in/out
r: reset view
R: reset display params
q: quit
f: toggle fps display
c: toggle colors display
u: toggle colors update
o: toggle adaptative opacity
s: toggle star display (pixels)
n: toggle nebulae display
t: track nearest star
T: disable tracking
p: pause (or <space>)
h: toggle help display""")

    def _fps(self):
        """ Update frame time and id and return the fps. """
        curr_time = time.time()
        duration = curr_time - self.frame_times[self.frame_id]
        self.frame_times[self.frame_id] = curr_time
        self.frame_id = (self.frame_id + 1) % len(self.frame_times)

        return len(self.frame_times) / duration

    def _print_fps(self):
        """ Calculate and print fps. """
        self._print("{:.1f}fps".format(self._fps()))

    def _find_nearest_star(self, x, y):
        """ Return the index of the nearest star from mouse coordinates. """
        mouse_pos = \
            self.axis.origin + self.axis.scale * np.asarray([x, self.size[1]-y])
        return ((self.simu.coords() - mouse_pos) ** 2).sum(axis=1).argmin()

    def _calc_opacity_factor(self):
        """ Returns the factor applied to the opacity """
        if self.use_adaptative_opacity:
            return max(0.01, self.__ao_factor/self.axis.scale)
        else:
            return 1.

    def _bind_star_vbo(self):
        """ Bind the vertex buffer object with star coordinates. """
        # Bind the vertex VBO
        self._star_vbo.bind()

        # Tell OpenGL that the VBO contains an array of vertices
        glEnableClientState(GL_VERTEX_ARRAY)

        # These vertices contain 2 double precision coordinates
        glVertexPointer(2, GL_DOUBLE, 0, None)

    def _draw_pixels(self, frame_buffer=0):
        """ Draw stars as pixels.

        frame_buffer: int
            OpenGL frame buffer id.
        """

        glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer)
        self._bind_star_vbo()

        # Choose appropriate shader program
        if self.use_adaptative_opacity:
            glUseProgram(self._ao_shader_program)
            glUniform1f(glGetUniformLocation(
                self._ao_shader_program, 'opacity_factor'),
                self._calc_opacity_factor())
        else:
            glUseProgram(0)

        # Draw "count" points from the VBO
        glDrawArrays(GL_POINTS, 0, self._star_count)

    def _draw_nebulae(self, frame_buffer=0):
        """ Draw stars as nebulae.

        frame_buffer: int
            OpenGL frame buffer id.
        """

        glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer)
        self._bind_star_vbo()

        # Enabling sprite and point size
        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_POINT_SPRITE)

        # Loading shader program and binding variables
        glUseProgram(self._nebulae_shader_program)

        def bind_uniform_1f(name, value):
            glUniform1f(glGetUniformLocation(
                self._nebulae_shader_program, name), value)

        bind_uniform_1f('opacity_factor', self._calc_opacity_factor())
        bind_uniform_1f('view_scale', self.axis.scale)
        bind_uniform_1f('radius_factor', self.nebulae_radius_factor)
        bind_uniform_1f('max_nebulae_radius', self.nebulae_max_radius)
        bind_uniform_1f('star_radius', self.star_radius)
        bind_uniform_1f('min_pixel_size', self.star_min_pixel_size)
        bind_uniform_1f('density_factor', self.nebulae_density_factor)

        # Draw "count" point sprites from the VBO
        glDrawArrays(GL_POINTS, 0, self._star_count)

        # Disabling sprite and point size
        glDisable(GL_PROGRAM_POINT_SIZE)
        glDisable(GL_POINT_SPRITE)

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
        glOrtho(self.axis.origin[0],
                self.axis.origin[0] + self.axis.scale * self.size[0],
                self.axis.origin[1],
                self.axis.origin[1] + self.axis.scale * self.size[1],
                -1, 1)

        # Background color
        glClearColor(0., 0., 0., 0.)

        # Draw color
        glColor(1., 1., 1.)

        # Draw stars as nebulae
        if self.use_nebulae_render:
            self._draw_nebulae()

        # Draw stars as pixels
        if self.use_pixel_render:
            self._draw_pixels()

        # Printing fps
        if self.use_fps:
            self._print_fps()

        # Printing help
        if self.use_help:
            self._print_help()

        # Swap display buffers
        glutSwapBuffers()


###############################################################################
# Demo
if __name__ == '__main__':
    """ Demo """

    class SpinningCloud:
        def __init__(self, size, theta=math.pi/18000):
            self._coords = np.random.randn(size, 2)
            self._colors = np.random.rand(size, 4)
            self._rot = np.asarray([[math.cos(theta), math.sin(theta)],
                                    [-math.sin(theta), math.cos(theta)]])

        def next(self):
            self._coords = np.dot(self._coords, self._rot)

        def coords(self):
            return self._coords

        def colors(self):
            return self._colors

    simu = SpinningCloud(100000, math.pi/1800)

    anim = Animation(simu, axis=[-2, 2, -2, 2], size=[480, 480])
    anim.use_colors = True
    anim.use_colors_update = False
    anim.use_adaptative_opacity = True
    anim.use_help = True

    anim.main_loop()
