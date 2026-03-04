import sys
print("1 - starting")
from minitorch import tensor
print("2 - tensor imported")
from minitorch.fast_ops import FastOps
print("3 - FastOps imported")
import minitorch.datasets as d
print("4 - datasets imported")
print(d.simple(5))
print("5 - all done")
