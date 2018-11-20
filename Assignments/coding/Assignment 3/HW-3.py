import os
import argparse
import numpy as np
from collections import namedtuple


# A convinient Tree structure
# Example Declearation:
## myTree = Tree(0.5, 0x---, 0x---, 0x---)
## Or
## myTree = Tree(Data1=0.5, Parent=0x---, Left=0x---, Right=0x---)
Tree = namedtuple("Tree", "Data Parent Left Right")
myTree = Tree(Data=0.5, Parent=None, Left=None, Right=None)
print(type(myTree))
