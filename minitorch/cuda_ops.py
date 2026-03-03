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

def _broadcast_shape(shape1, shape2):
    res = []
    for d1, d2 in zip(reversed(shape1), reversed(shape2)):
        if d1 != 1 and d2 != 1 and d1 != d2:
            raise Exception(f"Shapes {shape1} and {shape2} are not broadcastable")
        res.append(max(d1, d2))
    while len(shape1) < len(shape2):
        res.append(shape2[len(shape2) - len(shape1) - 1])
        shape1 = (1,) + shape1
    while len(shape2) < len(shape1):
        res.append(shape1[len(shape1) - len(shape2) - 1])
        shape2 = (1,) + shape2
    return tuple(reversed(res))

# =============================================================================
# KERNELS (GPU Side)
# =============================================================================

@cuda.jit(device=True)
def _device_to_index(ordinal, shape, out_index):
    to_index(ordinal, shape, out_index)

@cuda.jit(device=True)
def _device_broadcast_index(out_index, out_shape, in_shape, in_index):
    broadcast_index(out_index, out_shape, in_shape, in_index)

@cuda.jit(device=True)
def _device_index_to_position(index, strides):
    return index_to_position(index, strides)

def tensor_map(fn):
    f = cuda.jit(device=True)(fn)

    @cuda.jit()
    def _map(out, out_shape, out_strides, out_size, in_storage, in_shape, in_strides):
        i = cuda.grid(1)
        if i < out_size:
            out_index = cuda.local.array(MAX_DIMS, numba.int32)
            in_index = cuda.local.array(MAX_DIMS, numba.int32)
            _device_to_index(i, out_shape, out_index)
            _device_broadcast_index(out_index, out_shape, in_shape, in_index)
            o = _device_index_to_position(out_index, out_strides)
            j = _device_index_to_position(in_index, in_strides)
            out[o] = f(in_storage[j])

    def _launcher(out, in_t):
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (out.size + threadsperblock - 1) // threadsperblock
        _map[blockspergrid, threadsperblock](
            out._tensor._storage, out._tensor._shape, out._tensor._strides, out.size,
            in_t._tensor._storage, in_t._tensor._shape, in_t._tensor._strides
        )
    return _launcher

def tensor_zip(fn):
    f = cuda.jit(device=True)(fn)

    @cuda.jit()
    def _zip(out, out_shape, out_strides, out_size, a_storage, a_shape, a_strides, b_storage, b_shape, b_strides):
        i = cuda.grid(1)
        if i < out_size:
            out_index = cuda.local.array(MAX_DIMS, numba.int32)
            a_index = cuda.local.array(MAX_DIMS, numba.int32)
            b_index = cuda.local.array(MAX_DIMS, numba.int32)
            _device_to_index(i, out_shape, out_index)
            _device_broadcast_index(out_index, out_shape, a_shape, a_index)
            _device_broadcast_index(out_index, out_shape, b_shape, b_index)
            o = _device_index_to_position(out_index, out_strides)
            j = _device_index_to_position(a_index, a_strides)
            k = _device_index_to_position(b_index, b_strides)
            out[o] = f(a_storage[j], b_storage[k])

    def _launcher(out, a, b):
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (out.size + threadsperblock - 1) // threadsperblock
        _zip[blockspergrid, threadsperblock](
            out._tensor._storage, out._tensor._shape, out._tensor._strides, out.size,
            a._tensor._storage, a._tensor._shape, a._tensor._strides,
            b._tensor._storage, b._tensor._shape, b._tensor._strides
        )
    return _launcher

def tensor_reduce(fn, reduce_value):
    f = cuda.jit(device=True)(fn)

    @cuda.jit()
    def _reduce(out, out_shape, out_strides, out_size, a_storage, a_shape, a_strides, reduce_dim, reduce_val):
        i = cuda.grid(1)
        if i < out_size:
            out_index = cuda.local.array(MAX_DIMS, numba.int32)
            _device_to_index(i, out_shape, out_index)
            o = _device_index_to_position(out_index, out_strides)
            
            acc = reduce_val
            for s in range(a_shape[reduce_dim]):
                out_index[reduce_dim] = s
                j = _device_index_to_position(out_index, a_strides)
                acc = f(acc, a_storage[j])
            out[o] = acc

    def _launcher(out, a, dim):
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (out.size + threadsperblock - 1) // threadsperblock
        _reduce[blockspergrid, threadsperblock](
            out._tensor._storage, out._tensor._shape, out._tensor._strides, out.size,
            a._tensor._storage, a._tensor._shape, a._tensor._strides,
            dim, reduce_value
        )
    return _launcher

@cuda.jit()
def _mm_kernel(out, out_shape, out_strides, out_size, a_storage, a_shape, a_strides, b_storage, b_shape, b_strides):
    TILE = 32
    a_shared = cuda.shared.array((TILE, TILE), numba.float64)
    b_shared = cuda.shared.array((TILE, TILE), numba.float64)
    
    batch = cuda.blockIdx.z
    i = cuda.blockIdx.x * TILE + cuda.threadIdx.x
    j = cuda.blockIdx.y * TILE + cuda.threadIdx.y
    ti = cuda.threadIdx.x
    tj = cuda.threadIdx.y

    # Dimensions: A is (M x K), B is (K x L), Out is (M x L)
    M, K = a_shape[1], a_shape[2]
    L = b_shape[2]

    acc = 0.0
    for k_start in range(0, K, TILE):
        # 1. Load A into shared memory
        if i < M and (k_start + tj) < K:
            a_shared[ti, tj] = a_storage[batch * a_strides[0] + i * a_strides[1] + (k_start + tj) * a_strides[2]]
        else:
            a_shared[ti, tj] = 0.0

        # 2. Load B into shared memory
        if (k_start + ti) < K and j < L:
            b_shared[ti, tj] = b_storage[batch * b_strides[0] + (k_start + ti) * b_strides[1] + j * b_strides[2]]
        else:
            b_shared[ti, tj] = 0.0

        cuda.syncthreads()

        # 3. Compute partial dot product
        for k_offset in range(TILE):
            if (k_start + k_offset) < K:
                acc += a_shared[ti, k_offset] * b_shared[k_offset, tj]
        
        cuda.syncthreads()

    # 4. Write result to global memory
    if i < M and j < L:
        out[batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]] = acc

# =============================================================================
# LAUNCHERS
# =============================================================================

def matrix_multiply_launcher(out, a, b):
    TILE = 32
    threadsperblock = (TILE, TILE, 1)
    
    # 1. Get internal data
    out_data = out._tensor
    a_data = a._tensor
    b_data = b._tensor

    # 2. FORCE shapes to 3D tuples (Batch, Rows, Cols)
    # This ensures a_shape[2] NEVER causes an IndexError or Illegal Address
    def ensure_3d(s):
        if len(s) == 3: return s
        if len(s) == 2: return (1, s[0], s[1])
        return (1, 1, s[0])

    s_out = ensure_3d(out_data.shape)
    s_a = ensure_3d(a_data.shape)
    s_b = ensure_3d(b_data.shape)

    # 3. Calculate grid using guaranteed 3D dimensions
    blockspergrid = (
        (s_out[1] + TILE - 1) // TILE,
        (s_out[2] + TILE - 1) // TILE,
        s_out[0]
    )
    
    # 4. PASS THE 3D SHAPES (s_out, s_a, s_b) to the kernel, NOT the raw .shape
    _mm_kernel[blockspergrid, threadsperblock](
        out_data._storage, s_out, out_data._strides, out.size,
        a_data._storage, s_a, a_data._strides,
        b_data._storage, s_b, b_data._strides
    )

# =============================================================================
# DISPATCHER CLASS
# =============================================================================

class CudaOps:
    @staticmethod
    def map(fn):
        f = tensor_map(fn)
        def ret(a):
            out = a.zeros(a.shape)
            f(out, a)
            return out
        return ret

    @staticmethod
    def zip(fn):
        f = tensor_zip(fn)
        def ret(a, b):
            out_shape = _broadcast_shape(a.shape, b.shape)
            out = a.zeros(out_shape)
            f(out, a, b)
            return out
        return ret

    @staticmethod
    def reduce(fn, start=0.0):
        f = tensor_reduce(fn, start)
        def ret(a, dim):
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.zeros(tuple(out_shape))
            f(out, a, dim)
            return out
        return ret

    @staticmethod
    def matrix_multiply(a, b):
        a_shape = a.shape
        b_shape = b.shape
        
        batch = a_shape[0] if len(a_shape) == 3 else 1
        m = a_shape[1] if len(a_shape) == 3 else a_shape[0]
        p = b_shape[2] if len(b_shape) == 3 else b_shape[1]
        
        out = a.zeros((batch, m, p))
        matrix_multiply_launcher(out, a, b)
        
        if len(a_shape) == 2 and len(b_shape) == 2:
            return out.view(m, p)
        return out

    cmap = map
    cuda = True