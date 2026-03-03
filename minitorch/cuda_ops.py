from numba import cuda
import numba
from .tensor_data import (
    MAX_DIMS,
    to_index,
    index_to_position,
    broadcast_index,
)

# Constants
THREADS_PER_BLOCK = 32

def tensor_map(fn):
    f = cuda.jit(device=True)(fn)

    @cuda.jit()
    def _map(out, out_shape, out_strides, out_size, in_storage, in_shape, in_strides):
        i = cuda.grid(1)
        if i < out_size:
            out_index = cuda.local.array(MAX_DIMS, numba.int32)
            in_index = cuda.local.array(MAX_DIMS, numba.int32)
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            o = index_to_position(out_index, out_strides)
            j = index_to_position(in_index, in_strides)
            out[o] = f(in_storage[j])

    return _map

def tensor_zip(fn):
    f = cuda.jit(device=True)(fn)

    @cuda.jit()
    def _zip(out, out_shape, out_strides, out_size, a_storage, a_shape, a_strides, b_storage, b_shape, b_strides):
        i = cuda.grid(1)
        if i < out_size:
            out_index = cuda.local.array(MAX_DIMS, numba.int32)
            a_index = cuda.local.array(MAX_DIMS, numba.int32)
            b_index = cuda.local.array(MAX_DIMS, numba.int32)
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            o = index_to_position(out_index, out_strides)
            j = index_to_position(a_index, a_strides)
            k = index_to_position(b_index, b_strides)
            out[o] = f(a_storage[j], b_storage[k])

    return _zip

    def tensor_reduce(fn):
    f = cuda.jit(device=True)(fn)

    @cuda.jit()
    def _reduce(out, out_shape, out_strides, out_size, a_storage, a_shape, a_strides, reduce_dim, reduce_value):
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.grid(1)
        local_i = cuda.threadIdx.x

        if i < out_size:
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            
            # Load initial value
            acc = reduce_value
            # Reduction loop over the target dimension
            for s in range(a_shape[reduce_dim]):
                out_index[reduce_dim] = s
                j = index_to_position(out_index, a_strides)
                acc = f(acc, a_storage[j])
            
            out[o] = acc
            
    return _reduce

    @cuda.jit()
def _tensor_matrix_multiply(out, out_shape, out_strides, out_size, a_storage, a_shape, a_strides, b_storage, b_shape, b_strides):
    # Fixed TILE size
    TILE = 32
    a_shared = cuda.shared.array((TILE, TILE), numba.float64)
    b_shared = cuda.shared.array((TILE, TILE), numba.float64)
    
    # Grid indexing
    batch = cuda.blockIdx.z
    i = cuda.blockIdx.x * TILE + cuda.threadIdx.x
    j = cuda.blockIdx.y * TILE + cuda.threadIdx.y
    
    # Local indexing within tile
    ti = cuda.threadIdx.x
    tj = cuda.threadIdx.y

    acc = 0.0
    for k_start in range(0, a_shape[2], TILE):
        # 1. Load A and B tiles into shared memory
        k = k_start + tj
        if i < a_shape[1] and k < a_shape[2]:
            a_shared[ti, tj] = a_storage[batch * a_strides[0] + i * a_strides[1] + k * a_strides[2]]
        else:
            a_shared[ti, tj] = 0.0

        k = k_start + ti
        if k < b_shape[1] and j < b_shape[2]:
            b_shared[ti, tj] = b_storage[batch * b_strides[0] + k * b_strides[1] + j * b_strides[2]]
        else:
            b_shared[ti, tj] = 0.0

        cuda.syncthreads()

        # 2. Multiply the loaded tiles
        for k_offset in range(TILE):
            if k_start + k_offset < a_shape[2]:
                acc += a_shared[ti, k_offset] * b_shared[k_offset, tj]
        
        cuda.syncthreads()

    # 3. Write final result
    if i < out_shape[1] and j < out_shape[2]:
        out[batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]] = acc

class CudaOps:
    map = tensor_map
    zip = tensor_zip
    reduce = tensor_reduce
    matrix_multiply = _tensor_matrix_multiply