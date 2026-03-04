import sys

print("1 - Starting import tests...")

# Test basic imports
from minitorch import Tensor, Scalar, Module, Parameter, SGD
print("2 - Core classes imported (Tensor, Scalar, Module, Parameter, SGD)")

# Test functions moved to nn.py
from minitorch import one_hot, no_grad, argmax
print("3 - one_hot, no_grad, argmax imported (moved from __init__.py to nn.py)")

# Test functions moved to testing.py
from minitorch import sum_practice, mm_practice
print("4 - sum_practice, mm_practice imported (moved from __init__.py to testing.py)")

# Test nn.py functions still work
from minitorch import avgpool2d, maxpool2d, softmax, logsoftmax, dropout, max
print("5 - NN functions imported (avgpool2d, maxpool2d, softmax, logsoftmax, dropout, max)")

# Test layer classes
from minitorch import Linear, Conv1d, Conv2d
print("6 - Layer classes imported (Linear, Conv1d, Conv2d)")

# Test other re-exports
from minitorch import central_difference, History, tensor, zeros, rand
print("7 - Autodiff and tensor functions imported")

from minitorch import TensorData, SimpleBackend, CudaOps
print("8 - Backends imported (SimpleBackend, CudaOps)")

from minitorch import MathTestVariable, grad_check
print("9 - Testing utilities imported")

# Verify they point to the right modules
assert one_hot.__module__ == "minitorch.nn", f"one_hot is in {one_hot.__module__}, expected minitorch.nn"
assert no_grad.__module__ == "minitorch.nn", f"no_grad is in {no_grad.__module__}, expected minitorch.nn"
assert argmax.__module__ == "minitorch.nn", f"argmax is in {argmax.__module__}, expected minitorch.nn"
assert sum_practice.__module__ == "minitorch.testing", f"sum_practice is in {sum_practice.__module__}, expected minitorch.testing"
assert mm_practice.__module__ == "minitorch.testing", f"mm_practice is in {mm_practice.__module__}, expected minitorch.testing"
print("10 - All module locations verified!")

print("\nAll imports passed! __init__.py is now a clean re-export file.")
