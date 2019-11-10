
import numpy as np


class FeModel:
    def __init__(self, nodes, elements):
        self.size = len(nodes) * 2
        self.nodes = nodes
        self.elements = elements
        self.k = np.zeros((self.size, self.size))
        self.assemble_global_stiffness_matrix()


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
        pass
