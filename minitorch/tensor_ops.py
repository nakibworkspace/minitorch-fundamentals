from __future__ import annotations
from typing import Callable, Optional, Type
import numpy as np

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
    Index, Shape, Storage, Strides,
)

def tensor_map(fn):
    """Higher-order tensor map."""
    def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):
        out_index = np.zeros(MAX_DIMS, dtype=np.int32)
        in_index = np.zeros(MAX_DIMS, dtype=np.int32)

        for i in range(int(np.prod(out_shape))):
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)

            out_pos = index_to_position(out_index, out_strides)
            in_pos = index_to_position(in_index[:len(in_shape)], in_strides)

            out[out_pos] = fn(in_storage[in_pos])

    return _map


def tensor_zip(fn):
    """Higher-order tensor zip."""
    def _zip(out, out_shape, out_strides,
             a_storage, a_shape, a_strides,
             b_storage, b_shape, b_strides):
        out_index = np.zeros(MAX_DIMS, dtype=np.int32)
        a_index = np.zeros(MAX_DIMS, dtype=np.int32)
        b_index = np.zeros(MAX_DIMS, dtype=np.int32)

        for i in range(int(np.prod(out_shape))):
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            out_pos = index_to_position(out_index, out_strides)
            a_pos = index_to_position(a_index[:len(a_shape)], a_strides)
            b_pos = index_to_position(b_index[:len(b_shape)], b_strides)

            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return _zip


def tensor_reduce(fn):
    """Higher-order tensor reduce."""
    def _reduce(out, out_shape, out_strides,
                a_storage, a_shape, a_strides, reduce_dim):
        out_index = np.zeros(MAX_DIMS, dtype=np.int32)
        a_index = np.zeros(MAX_DIMS, dtype=np.int32)

        for i in range(int(np.prod(out_shape))):
            to_index(i, out_shape, out_index)
            out_pos = index_to_position(out_index, out_strides)

            for j in range(len(out_shape)):
                a_index[j] = out_index[j]

            for j in range(a_shape[reduce_dim]):
                a_index[reduce_dim] = j
                a_pos = index_to_position(a_index[:len(a_shape)], a_strides)
                out[out_pos] = fn(out[out_pos], a_storage[a_pos])

    return _reduce