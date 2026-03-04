from typing import Tuple
from . import operators
from .tensor import Tensor
from .tensor_functions import Function, rand, zeros


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape for pooling."""
    batch, channel, height, width = input.shape
    kh, kw = kernel

    assert height % kh == 0
    assert width % kw == 0

    new_h = height // kh
    new_w = width // kw

    # Reshape to separate pooling regions
    x = input.contiguous().view(batch, channel, new_h, kh, new_w, kw)

    # Permute to group pooling dimensions at the end
    x = x.permute(0, 1, 2, 4, 3, 5)

    # Combine pooling dimensions
    x = x.contiguous().view(batch, channel, new_h, new_w, kh * kw)

    return x, new_h, new_w


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Average pooling."""
    batch, channel = input.shape[:2]
    tiled, new_h, new_w = tile(input, kernel)
    return tiled.mean(dim=4).contiguous().view(batch, channel, new_h, new_w)


from .fast_ops import FastOps

max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Return 1-hot tensor indicating maximum positions."""
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx, input: Tensor, dim: Tensor) -> Tensor:
        ctx.save_for_backward(input, int(dim.item()))
        return max_reduce(input, int(dim.item()))

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, float]:
        input, dim = ctx.saved_values
        return grad_output * argmax(input, dim), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    x_max = max(input, dim)
    x_stable = input - x_max
    exp_x = x_stable.exp()
    sum_exp = exp_x.sum(dim=dim)
    return exp_x / sum_exp


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    x_max = max(input, dim)
    x_stable = input - x_max
    log_sum_exp = x_stable.exp().sum(dim=dim).log()
    return x_stable - log_sum_exp


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    batch, channel = input.shape[:2]
    tiled, new_h, new_w = tile(input, kernel)
    return max(tiled, dim=4).contiguous().view(batch, channel, new_h, new_w)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    if ignore or rate == 0.0:
        return input
    if rate >= 1.0:
        return input * 0.0
    mask = rand(input.shape, backend=input.backend) > rate
    return input * mask / (1.0 - rate)


import math
from .module import Module, Parameter
from .tensor_functions import Conv1dFun, Conv2dFun
from .tensor_ops import TensorBackend
from .fast_ops import FastOps

FastBackend = TensorBackend(FastOps)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        k = 1.0 / math.sqrt(in_features)
        w = (rand((in_features, out_features), backend=FastBackend) * (2 * k) - k).detach()
        b = (rand((out_features,), backend=FastBackend) * (2 * k) - k).detach()
        w.requires_grad_(True)
        b.requires_grad_(True)
        self.weight = Parameter(w)
        self.bias = Parameter(b)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.value + self.bias.value


class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: Tuple[int, int]):
        super().__init__()
        kh, kw = kernel
        k = 1.0 / math.sqrt(in_channels * kh * kw)
        w = (rand((out_channels, in_channels, kh, kw), backend=FastBackend) * (2 * k) - k).detach()
        w.requires_grad_(True)
        self.weight = Parameter(w)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        return Conv2dFun.apply(x, self.weight.value)


class Conv1d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_width: int):
        super().__init__()
        k = 1.0 / math.sqrt(in_channels * kernel_width)
        w = (rand((out_channels, in_channels, kernel_width), backend=FastBackend) * (2 * k) - k).detach()
        w.requires_grad_(True)
        self.weight = Parameter(w)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        return Conv1dFun.apply(x, self.weight.value)

def one_hot(labels, num_classes):                                                                                                                                 
      """Convert integer labels to one-hot encoded tensor."""                                                                                                       
      n = labels.size                                                                                                                                               
      out = zeros((n, num_classes), backend=labels.backend)                                                                                                         
      for i in range(n):                                                                                                                                            
          label_idx = int(float(labels._tensor._storage[i]))                                                                                                        
          out._tensor._storage[i * num_classes + label_idx] = 1.0
      return out

class no_grad:
      """Context manager to disable gradient tracking."""
      def __enter__(self):
          return self

      def __exit__(self, *args):
          pass