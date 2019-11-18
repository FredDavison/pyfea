
import numpy as np


class FeModel:
    """Represents the global FE model and implements solution."""

    def __init__(self, nodes, elements, forces=None,
                 displacements=None):

        self.size = len(nodes) * 2
        self.nodes = nodes
        self.elements = elements
        self.forces = np.asarray(forces or [0 for i in range(self.size)])
        self.displacements = np.asarray(displacements or [None for i in range(self.size)])
        self.k = np.zeros((self.size, self.size))
        self.assemble_global_stiffness_matrix()

        self.reduced_k = None
        self.reduced_f = None
        self.reduced_u = None

    def assemble_global_stiffness_matrix(self):
        for element in self.elements:
            i = element.i_node.idx * 2
            j = element.j_node.idx * 2
            k = element.global_stiffness_matrix()

            self.k[i:i + 2, i:i + 2] += k[:2, :2]
            self.k[j:j + 2, i:i + 2] += k[2:, :2]

            self.k[i:i + 2, j:j + 2] += k[:2, 2:]
            self.k[j:j + 2, j:j + 2] += k[2:, 2:]

    def solve(self):
        unknown_displacement_idxs = np.argwhere(self.displacements != 0).ravel()
        self.reduced_k = self.k[unknown_displacement_idxs][:, unknown_displacement_idxs]
        self.reduced_f = self.forces[unknown_displacement_idxs]
        self.reduced_u = np.linalg.solve(self.reduced_k, self.reduced_f)
        self.displacements[unknown_displacement_idxs] = self.reduced_u
