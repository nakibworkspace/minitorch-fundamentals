from typing import Tuple
import numpy as np
from numba import njit, prange

from .tensor_data import (
    MAX_DIMS,
    Index,
    Shape,
    Strides,
    broadcast_index,
    index_to_position,
    to_index,
)

# JIT compile helper functions
to_index = njit(inline="always")(to_index.py_func)
index_to_position = njit(inline="always")(index_to_position.py_func)
broadcast_index = njit(inline="always")(broadcast_index.py_func)


def _tensor_conv1d(
    out, out_shape, out_strides, out_size,
    input, input_shape, input_strides,
    weight, weight_shape, weight_strides,
    reverse
):
    batch, out_channels, out_width = out_shape
    batch_, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    s1_batch, s1_chan, s1_w = input_strides[0], input_strides[1], input_strides[2]
    s2_out, s2_in, s2_k = weight_strides[0], weight_strides[1], weight_strides[2]
    so_batch, so_chan, so_w = out_strides[0], out_strides[1], out_strides[2]

    for b in prange(batch):
        for oc in range(out_channels):
            for w in range(out_width):
                out_pos = b * so_batch + oc * so_chan + w * so_w

                acc = 0.0
                for ic in range(in_channels):
                    for k in range(kw):
                        if reverse:
                            w_in = w - k
                        else:
                            w_in = w + k

                        if 0 <= w_in < width:
                            in_pos = b * s1_batch + ic * s1_chan + w_in * s1_w
                            weight_pos = oc * s2_out + ic * s2_in + k * s2_k
                            acc += input[in_pos] * weight[weight_pos]

                out[out_pos] = acc


tensor_conv1d = njit(parallel=True)(_tensor_conv1d)

def _tensor_conv2d(
    out, out_shape, out_strides, out_size,
    input, input_shape, input_strides,
    weight, weight_shape, weight_strides,
    reverse
):
    batch, out_channels, out_height, out_width = out_shape
    batch_, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    s1_batch, s1_chan, s1_h, s1_w = (
        input_strides[0], input_strides[1],
        input_strides[2], input_strides[3]
    )
    s2_out, s2_in, s2_kh, s2_kw = (
        weight_strides[0], weight_strides[1],
        weight_strides[2], weight_strides[3]
    )
    so_batch, so_chan, so_h, so_w = (
        out_strides[0], out_strides[1],
        out_strides[2], out_strides[3]
    )

    for b in prange(batch):
        for oc in range(out_channels):
            for h in range(out_height):
                for w in range(out_width):
                    out_pos = b * so_batch + oc * so_chan + h * so_h + w * so_w

                    acc = 0.0
                    for ic in range(in_channels):
                        for ki in range(kh):
                            for kj in range(kw):
                                if reverse:
                                    h_in = h - ki
                                    w_in = w - kj
                                else:
                                    h_in = h + ki
                                    w_in = w + kj

                                if 0 <= h_in < height and 0 <= w_in < width:
                                    in_pos = (b * s1_batch + ic * s1_chan +
                                              h_in * s1_h + w_in * s1_w)
                                    weight_pos = (oc * s2_out + ic * s2_in +
                                                  ki * s2_kh + kj * s2_kw)
                                    acc += input[in_pos] * weight[weight_pos]

                    out[out_pos] = acc


tensor_conv2d = njit(parallel=True, fastmath=True)(_tensor_conv2d)