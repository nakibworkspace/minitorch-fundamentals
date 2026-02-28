from __future__ import annotations
from typing import Iterable, Optional, Sequence, Tuple, Union
import numpy as np
import numpy.typing as npt
from numpy import array, float64

MAX_DIMS = 32

class IndexingError(RuntimeError):
    """Exception raised for indexing errors."""
    pass

# Type aliases for clarity
Storage = npt.NDArray[np.float64]
OutIndex = npt.NDArray[np.int32]
Index = npt.NDArray[np.int32]
Shape = npt.NDArray[np.int32]
Strides = npt.NDArray[np.int32]
UserIndex = Sequence[int]
UserShape = Sequence[int]
UserStrides = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """Convert multidimensional index to storage position."""
    position = 0
    for i, s in zip(index, strides):
        position += i * s
    return int(position)


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """Convert ordinal to multidimensional index."""
    cur_ord = ordinal
    for i in range(len(shape) - 1, -1, -1):
        out_index[i] = cur_ord % shape[i]
        cur_ord = cur_ord // shape[i]

class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")

        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def permute(self, *order: int) -> TensorData:
        """Permute tensor dimensions."""
    assert list(sorted(order)) == list(range(len(self.shape)))

    new_shape = tuple(self.shape[o] for o in order)
    new_strides = tuple(self.strides[o] for o in order)

    return TensorData(self._storage, new_shape, new_strides)

    def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """Broadcast two shapes to create a new union shape."""
    result = []
    len1, len2 = len(shape1), len(shape2)
    max_len = max(len1, len2)

    for i in range(max_len):
        d1 = shape1[len1 - 1 - i] if i < len1 else 1
        d2 = shape2[len2 - 1 - i] if i < len2 else 1

        if d1 == d2:
            result.append(d1)
        elif d1 == 1:
            result.append(d2)
        elif d2 == 1:
            result.append(d1)
        else:
            raise IndexingError(f"Cannot broadcast shapes {shape1} and {shape2}")

    return tuple(reversed(result))


    def broadcast_index(
        big_index: Index,
        big_shape: Shape,
        shape: Shape,
        out_index: OutIndex
    ) -> None:
        """Convert index from broadcasted shape to original shape."""
        offset = len(big_shape) - len(shape)

        for i in range(len(shape)):
            if shape[i] == 1:
                out_index[i] = 0
            else:
                out_index[i] = big_index[i + offset]