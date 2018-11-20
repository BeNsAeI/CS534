import os                                                                                                      >> 0.0000 -> 0.0001
import argparse                                                                                                >> 0.0000 -> 0.0003
import numpy as np
from collections import namedtuple


# A convinient Tree structure
# Example Declearation:
## myTree = Tree(0.5, 0x---, 0x---, 0x---)
## Or
## myTree = Tree(Data1="0.5, Parent=0x---, Left=0x---, Right=0x---)
Tree = namedtuple("Tree", "Data Parent Left Right")

