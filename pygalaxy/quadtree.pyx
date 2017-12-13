""" A Cython implementation of a QuadTree and other useful classes"""
import numpy as np
cimport numpy as np

# Numeric type
NUM_DTYPE = np.double
# Correspondant C type
ctypedef np.double_t NUM_DTYPE_T

cdef class Vector2D:
    """ A thin wrapper around numpy array to represent a 2D vector

    Args:
    coords(np.ndarray): A 2-element numpy array representing the coordinates of
        the vector

    """

    cdef np.ndarray _coords

    def __cinit__(self, np.ndarray coords):
        self._coords = coords

    # === CYTHON Get/Set ====
    cdef NUM_DTYPE_T get_x(self):
        return self._coords[0]

    cdef NUM_DTYPE_T get_y(self):
        return self._coords[1]

    cdef void set_x(self, NUM_DTYPE_T value):
        self._coords[0] = value

    cdef void set_y(self, NUM_DTYPE_T value):
        self._coords[1] = value

    cdef np.ndarray as_array(self):
        return self._coords

cdef class Body:
    """ A body to be inserted in the tree

    Args:
        - pos(np.ndarray): The position vector of the Body
        - vel(np.ndarray): The velocity vector of the Body
        - mass(float): The mass of the Body

    """

    # Position
    cdef Vector2D _pos

    # Velocity
    cdef Vector2D _vel

    # Mass
    cdef public NUM_DTYPE_T mass

    # Force acting on this body
    cdef public np.ndarray force

    def __cinit__(self, np.ndarray pos, np.ndarray vel, mass):
        self._pos = Vector2D(pos)
        self._vel = Vector2D(vel)
        self.mass = mass

    # === Python Get/Set ===

    @property
    def x(self):
        self._pos.get_x()

    @x.setter
    def x(self, value):
        self._pos.set_x(value)

    @property
    def y(self):
        self._pos.get_y()

    @y.setter
    def y(self, value):
        self._pos.set_y(value)

    @property
    def vx(self):
        self._vel.get_x()

    @vx.setter
    def vx(self, value):
        self._vel.set_x(value)

    @property
    def vy(self):
        self._vel.get_y()

    @vy.setter
    def vy(self, value):
        self._vel.set_y(value)

    cpdef np.ndarray pos(self):
        """ Returns the position of the body as 2D numpy array """
        return self._pos.as_array()

    cpdef np.ndarray vel(self):
        """ Returns the velocity of the body as 2D numpy array """
        return self._vel.as_array()

cdef class TreeNode:
    """ A node of the tree

    Args:
        bodies(list): A list of bodies to be inserted in the tree

    """

    # Function pointer for callback in TreeNode traverse
    ctypedef bool (*traverse_func)(TreeNode)

    # == Private Attributes ==
    cdef np.ndarray quads

    # Bottom-Left and Top-Right position vectors of the TreeNode bounds
    cdef Vector2D _bl, _tl

    # === Public Attributes ===
    # cdef public Body body

    # Total mass of the TreeNode
    # cdef public NUM_DTYPE_T total_mass

    # Center of gravity vector
    # cdef public np.ndarray cg

    # === Handy accessors ===

    cdef TreeNode* ne(self):
        if self._quads:
            return self._quads[0]
        else:
            return NULL

    cdef TreeNode* nw(self):
        if self._quads:
            return self._quads[1]
        else:
            return NULL

    cdef TreeNode* sw(self):
        if self._quads:
            return self._quads[2]
        else:
            return NULL

    cdef TreeNode* se(self):
        if self._quads:
            return self._quads[3]
        else:
            return NULL


    cpdef traverse (self, traverse_func should_visit):
        """ This method traverse recursively the Node if the function
        `should_visit`, applied to the node, returns True

        Args:
        should_visit(traverse_func): A function that returns True if the node
            must be traversed furtherly or not

        Yields:
            quad(TreeNode): A leaf TreeNode
        """

        for quad in self._quads:
            if should_visit(quad):
                yield quad.traverse(should_visit)
            else:
                yield quad

cdef class QTree:
    cdef TreeNode root


    def __cinit__(self, np.ndarray bodies):

        # The bottom leftest and top rightest points of the root quadrant
        cdef np.ndarray bot_left = Vector2D(np.zeros(2, 0, dtype=NUM_DTYPE))
        cdef np.ndarray top_right = Vector2D(np.zeros(2, 0, dtype=NUM_DTYPE))

        for body in bodies:
            cdef NUM_DTYPE_T body_x = body.pos.x
            cdef NUM_DTYPE_T body_y = body.pos.y

            if body_x < bot_left.x:
                bot_left.set_x(body_x)

            if body_y < bot_left.y:
                bot_left.set_y(body_y)

            if body_x > top_right.x:
                top_right.set_x(body_x)

            if body_y > top_right.y:
                top_right.set_y(body_y)

            self.add(body)

    cpdef add(self, Body body):
        pass
