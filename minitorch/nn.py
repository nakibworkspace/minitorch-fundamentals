from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


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
    tiled, new_h, new_w = tile(input, kernel)
    return tiled.mean(dim=4)


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input : input tensor
        dim : dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
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
    tiled, new_h, new_w = tile(input, kernel)
    return max(tiled, dim=4)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    if ignore:
        return input
    mask = rand(input.shape, backend=input.backend) > rate
    return input * mask / (1 - rate)