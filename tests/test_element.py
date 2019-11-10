"""Test properties of elements.

Reference values are mostly taken from 'Introduction to Finite Element
Methods' by Carlos A Felippa."""

import pytest

import numpy as np

from pyfea.elements import Node, Bar


nodes = [Node([0, 0]),
         Node([1, 0]),
         Node([1, 1])]

node_pairs = [(nodes[0], nodes[1]),
              (nodes[1], nodes[2]),
              (nodes[0], nodes[2])]

reference_ks = [
    [[1, 0, -1, 0],
     [0, 0, 0, 0],
     [-1, 0, 1, 0],
     [0, 0, 0, 0]],
    [[0, 0, 0, 0],
     [0, 1, 0, -1],
     [0, 0, 0, 0],
     [0, -1, 0, 1]],
    [[0.5, 0.5, -0.5, -0.5],
     [0.5, 0.5, -0.5, -0.5],
     [-0.5, -0.5, 0.5, 0.5],
     [-0.5, -0.5, 0.5, 0.5]]
]


@pytest.mark.parametrize("node_pair, reference", zip(node_pairs, reference_ks))
def test_bar_global_stiffness_matrix(node_pair, reference):
    bar = Bar(*node_pair, diameter=1)
    k_coefficients = bar.global_stiffness_matrix() / bar.axial_rigidity
    np.testing.assert_array_almost_equal(k_coefficients, reference)
