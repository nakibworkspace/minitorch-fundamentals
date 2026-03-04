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
from .tensor_functions import Conv1dFun, Conv2dFun
from .nn import avgpool2d, maxpool2d, softmax, logsoftmax, dropout, max, Linear, Conv1d, Conv2d
from .fast_ops import *
from .tensor_ops import *
from .cuda_ops import CudaOps
from .cuda_ops import sum_practice as _cuda_sum_practice

def sum_practice(a):
    # ._tensor is the TensorData object that actually has the ._storage
    return _cuda_sum_practice(a)._tensor

def mm_practice(a, b):
    # Return the internal _tensor so the tests can find ._storage
    return a.f.matrix_multiply(a, b)._tensor

def one_hot(labels, num_classes):
    """Convert integer labels to one-hot encoded tensor."""
    n = labels.size
    out = zeros((n, num_classes), backend=labels.backend)
    for i in range(n):
        label_idx = int(float(labels._tensor._storage[i]))
        out._tensor._storage[i * num_classes + int(labels._tensor._storage[i])] = 1.0
    return out


class no_grad:
    """Context manager to disable gradient tracking."""
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def argmax(t, dim):
    """Return indices of max values along dim."""
    m = max(t, dim)
    mask = (m == t)
    indices = []
    for i in range(t.shape[dim]):
        indices.append(float(i))
    idx_shape = [1] * len(t.shape)
    idx_shape[dim] = t.shape[dim]
    idx_tensor = Tensor.make(indices, tuple(idx_shape), backend=t.backend)
    return (mask * idx_tensor).sum(dim)

