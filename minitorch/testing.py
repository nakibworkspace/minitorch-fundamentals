from minitorch.operators import map, zipWith, reduce, mul, add

# Dot product of two vectors
def dot(a, b):
    return reduce(add, 0)(zipWith(mul)(a, b))

print(dot([1, 2, 3], [4, 5, 6]))

def assert_close(a: float, b: float) -> None:
    if abs(a - b) > 1e-2:
        raise AssertionError(f"{a} and {b} are not close enough")