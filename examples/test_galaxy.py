#!/usr/bin/env python

"""
This program uses the Barnes-Hut algorithm to solve the N-body problem on the
galaxy.


Usage:
    solar [options]

Options:
    -R, --render=<render_option>    The typology of render engine to be used. By
                                    default it uses `matplotlib`. The other
                                    option is to use the more fancy `opengl`.
                                    [default: matplotlib]

    --step=<step>                   Simulation step between each render
                                    [default: 5]
"""
import numpy as np
import importlib

from docopt import docopt

import pygalaxy
from pygalaxy.barnes_hut_array import compute_energy


def temp2color(temps):
    """ Convert black body temperature to corresponding RGBA color.

    See http://www.tannerhelland.com/4435/convert-temperature-rgb-algorithm-code/
    """
    colors = np.empty((temps.size, 4))
    scaled_temps = temps/100.

    # Red component
    ind = scaled_temps <= 66.
    colors[ind, 0] = 255.
    colors[~ind, 0] = 329.698727446 * ((scaled_temps[~ind] - 60.) ** -0.1332047592)

    # Green component
    colors[ind, 1] = 99.4708025861 * np.log(scaled_temps[ind]) - 161.1195681661
    colors[~ind, 1] = 288.1221695283 * ((scaled_temps[~ind] - 60.) ** -0.0755148492)

    # Blue component
    ind = scaled_temps >= 66.
    colors[ind, 2] = 255.
    colors[~ind, 2] = 138.5177312231 * np.log(scaled_temps[~ind] - 10.) - 305.0447927307
    colors[scaled_temps <= 19., 2] = 0.

    # Alpha component
    colors[:, 3] = 255.

    # Clipping values
    np.clip(colors/255., 0., 1., out=colors)

    return colors


class Galaxy:
    def __init__(self, blackHole, dt=10., display_step=1):
        self.mass, self.particles = pygalaxy.init_collisions(blackHole)
        # self.time_method = pygalaxy.ADB6(dt, self.particles.shape[0], compute_energy)
        # self.time_method = pygalaxy.Euler_symplectic(dt, self.particles.shape[0], compute_energy)
        # self.time_method = pygalaxy.Stormer_verlet(dt, self.particles.shape[0], compute_energy)
        self.time_method = pygalaxy.Optimized_815(
            dt, self.particles.shape[0], compute_energy)
        self.display_step = display_step
        self.it = 0

    def next(self):
        for i in range(self.display_step):
            self.it += 1
            print(self.it)
            self.time_method.update(self.mass, self.particles)

    def coords(self):
        return self.particles[:, :2]

    def colors(self):
        speed_magnitude = np.linalg.norm(self.particles[:, 2:4], axis=1)
        speed_min = speed_magnitude.min()
        speed_max = speed_magnitude.max()
        colors = temp2color(
            3000 + 6000*(speed_magnitude-speed_min)/(speed_max - speed_min))
        colors[:, 3] = 0.05

        return colors + np.asarray([0., 0., 0., 0.95]) * \
            np.minimum(self.mass, 20).reshape(-1, 1) / 20


if __name__ == '__main__':
    args = docopt(__doc__)

    display_step = int(args['--step'])
    render_engine = args['--render']

    # Importing the right class for rendering from the right module
    anim_module = importlib.import_module('pygalaxy.'+render_engine)
    Animation = getattr(anim_module, 'Animation')

    np.random.seed(42)

    blackHole = [
        {
            'coord': [0, 0],
            'mass': 1000000,
            'svel': 1,
            'stars': 2000,
            'radstars': 3
        },

        {
            'coord': [3, 3],
            'mass': 1000000,
            'svel': 0.9,
            'stars': 1000,
            'radstars': 1
        }]

    sim = Galaxy(blackHole, display_step=display_step)

    anim = Animation(sim, axis=[-10., 10., -10., 10.])

    if args['--render'] == 'opengl':
        anim.use_colors = True
        anim.use_colors_update = False
        anim.use_adaptative_opacity = True
        anim.use_pixel_render = False
        anim.use_nebulae_render = True

    anim.main_loop()
