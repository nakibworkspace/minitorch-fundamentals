from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"

        # This line JIT compiles your tensor_map
        f = tensor_map(njit()(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        "See `tensor_ops.py`"

        f = tensor_zip(njit()(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        "See `tensor_ops.py`"
        f = tensor_reduce(njit()(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """
        Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
            a : tensor data a
            b : tensor data b

        Returns:
            New tensor data
        """

        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(fn):
    def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):
        stride_aligned = (
            len(out_shape) == len(in_shape) and
            np.array_equal(out_shape, in_shape) and
            np.array_equal(out_strides, in_strides)
        )

        if stride_aligned:
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])
        else:
            for i in prange(int(np.prod(out_shape))):
                out_index = np.zeros(MAX_DIMS, dtype=np.int32)
                in_index = np.zeros(MAX_DIMS, dtype=np.int32)

                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)

                out_pos = index_to_position(out_index, out_strides)
                in_pos = index_to_position(in_index[:len(in_shape)], in_strides)

                out[out_pos] = fn(in_storage[in_pos])

    return njit(parallel=True)(_map)


def tensor_zip(fn):
    def _zip(out, out_shape, out_strides,
             a_storage, a_shape, a_strides,
             b_storage, b_shape, b_strides):
        stride_aligned = (
            len(out_shape) == len(a_shape) == len(b_shape) and
            np.array_equal(out_shape, a_shape) and
            np.array_equal(out_shape, b_shape) and
            np.array_equal(out_strides, a_strides) and
            np.array_equal(out_strides, b_strides)
        )

        if stride_aligned:
            for i in prange(len(out)):
                out[i] = fn(a_storage[i], b_storage[i])
        else:
            for i in prange(int(np.prod(out_shape))):
                out_index = np.zeros(MAX_DIMS, dtype=np.int32)
                a_index = np.zeros(MAX_DIMS, dtype=np.int32)
                b_index = np.zeros(MAX_DIMS, dtype=np.int32)

                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, a_shape, a_index)
                broadcast_index(out_index, out_shape, b_shape, b_index)

                out_pos = index_to_position(out_index, out_strides)
                a_pos = index_to_position(a_index[:len(a_shape)], a_strides)
                b_pos = index_to_position(b_index[:len(b_shape)], b_strides)

                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return njit(parallel=True)(_zip)


def tensor_reduce(fn):
    def _reduce(out, out_shape, out_strides,
                a_storage, a_shape, a_strides, reduce_dim):
        for i in prange(int(np.prod(out_shape))):
            out_index = np.zeros(MAX_DIMS, dtype=np.int32)
            to_index(i, out_shape, out_index)
            out_pos = index_to_position(out_index, out_strides)

            a_index = out_index.copy()

            for j in range(a_shape[reduce_dim]):
                a_index[reduce_dim] = j
                a_pos = index_to_position(a_index[:len(a_shape)], a_strides)
                out[out_pos] = fn(out[out_pos], a_storage[a_pos])

    return njit(parallel=True)(_reduce)

def _tensor_matrix_multiply(
    out, out_shape, out_strides,
    a_storage, a_shape, a_strides,
    b_storage, b_shape, b_strides
):
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    batch_size = out_shape[0]
    M = out_shape[1]
    N = out_shape[2]
    K = a_shape[2]

    a_row_stride = a_strides[1]
    a_col_stride = a_strides[2]
    b_row_stride = b_strides[1]
    b_col_stride = b_strides[2]
    out_batch_stride = out_strides[0]
    out_row_stride = out_strides[1]
    out_col_stride = out_strides[2]

    for n in prange(batch_size):
        for i in range(M):
            for j in range(N):
                out_pos = (
                    n * out_batch_stride +
                    i * out_row_stride +
                    j * out_col_stride
                )

                acc = 0.0
                for k in range(K):
                    a_pos = (
                        n * a_batch_stride +
                        i * a_row_stride +
                        k * a_col_stride
                    )
                    b_pos = (
                        n * b_batch_stride +
                        k * b_row_stride +
                        j * b_col_stride
                    )
                    acc += a_storage[a_pos] * b_storage[b_pos]

                out[out_pos] = acc


tensor_matrix_multiply = njit(parallel=True, fastmath=True)(_tensor_matrix_multiply)