
from math import pi, sin, cos, acos

import numpy as np


E = 2.05e10
RHO = 7850


class Node:
    """Finite element node"""

    def __init__(self, coords, index):
        self.idx = index
        self.coords = np.asarray(coords)


class Bar:
    """Solid bar element"""

    def __init__(self, i_node, j_node, diameter):
        self.area = pi * diameter ** 2 / 4
        self.i_node = i_node
        self.j_node = j_node
        self.vector = self.j_node.coords - self.i_node.coords
        self.length = np.linalg.norm(self.vector)
        self.axial_rigidity = self.area * E / self.length
        self.orientation_angle = acos(self.vector[0] / self.length)
        self._s = sin(self.orientation_angle)
        self._c = cos(self.orientation_angle)

    @property
    def mass(self):
        return self.area * self.length * RHO

    def global_stiffness_matrix(self):
        Kbar = self.local_stiffness_matrix()
        T = self.transformation_matrix()
        TT = self.force_transformation_matrix()
                      
        K = np.linalg.multi_dot([TT, Kbar, T])
        return K

    def local_stiffness_matrix(self):
        coefficients = np.array([[ 1, 0, -1, 0],
                                 [ 0, 0,  0, 0],
                                 [-1, 0,  1, 0],
                                 [ 0, 0,  0, 0]])

        return self.axial_rigidity * coefficients

    def transformation_matrix(self):
        s, c = self._s, self._c
        return np.array([[ c,  s,  0,  0],
                         [-s,  c,  0,  0],
                         [ 0,  0,  c,  s],
                         [ 0,  0, -s,  c]])
    
    def force_transformation_matrix(self):
        s, c = self._s, self._c
        return np.array([[c, -s, 0,  0],
                         [s,  c, 0,  0],
                         [0,  0, c, -s],
                         [0,  0, s,  c]])
