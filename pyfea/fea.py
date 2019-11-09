
import numpy as np

from pyfea.elements import Bar


NODES = [(0, 0),
         (1, 0),
         (1, 1)]

MEMBERS = [(0, 1),
           (1, 2),
           (0, 2)]

# Support list twinned with nodes list.
# Tuple of x, y fixity.
# 1 indicates fixed.
SUPPORTS = [(1, 1),
            (0, 1),
            (0, 0)]

elements = []

for mbr in MEMBERS:
    end_coords = NODES[mbr[0]], NODES[mbr[1]]
    length = np.linalg.norm(end_coords)
    elements.append(Bar(*end_coords, diameter=0.25))

import ipdb; ipdb.set_trace()
