from .autodiff import central_difference
from .scalar import Scalar
from . import operators
from .autodiff import central_difference, History
from .scalar import Scalar
from .tensor_data import TensorData, IndexingError, shape_broadcast
from .operators import prod
from .tensor import Tensor
from .tensor_functions import tensor, zeros, rand
from .optim import SGD
from .testing import MathTestVariable, grad_check
from .tensor_ops import SimpleBackend
from .module import Module, Parameter
from .tensor_functions import Conv1dFun
from .fast_ops import *
from .tensor_ops import *
from .cuda_ops import CudaOps

def sum_practice(a):
    # ._tensor is the TensorData object that actually has the ._storage
    return a.f.add_reduce(a, 0)._tensor

def mm_practice(a, b):
    # Return the internal _tensor so the tests can find ._storage
    return a.f.matrix_multiply(a, b)._tensor
