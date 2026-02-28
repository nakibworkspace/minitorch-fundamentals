from minitorch.operators import map, zipWith, reduce, mul, add

# Dot product of two vectors
def dot(a, b):
    return reduce(add, 0)(zipWith(mul)(a, b))

print(dot([1, 2, 3], [4, 5, 6]))

def assert_close(a: float, b: float) -> None:
    if abs(a - b) > 1e-2:
        raise AssertionError(f"{a} and {b} are not close enough")

from minitorch.autodiff import central_difference

def square(x):
    return x * x

def mul(x, y):
    return x * y

print(central_difference(square, 3.0))           # Derivative of x² at x=3
print(central_difference(mul, 3.0, 4.0, arg=0))  # ∂(xy)/∂x at (3,4)
print(central_difference(mul, 3.0, 4.0, arg=1))  # ∂(xy)/∂y at (3,4)

from minitorch.scalar import Scalar

a = Scalar(2.0, name="a")
b = Scalar(3.0, name="b")

print(f"a.data = {a.data}")
print(f"a.is_leaf() = {a.is_leaf()}")
print(f"a.derivative = {a.derivative}")

a.accumulate_derivative(1.0)
a.accumulate_derivative(2.0)
print(f"After accumulating: a.derivative = {a.derivative}")


from minitorch.scalar import Scalar

a = Scalar(2.0)
a.requires_grad_(True)
b = Scalar(3.0)
b.requires_grad_(True)

c = a * b  # c = 6.0

print(f"c.data = {c.data}")
print(f"c.is_leaf() = {c.is_leaf()}")
print(f"c.history.last_fn = {c.history.last_fn}")

from minitorch.scalar import Scalar

a = Scalar(2.0)
a.requires_grad_(True)
b = Scalar(3.0)
b.requires_grad_(True)

c = a * b     # c = 6
d = c + a     # d = 6 + 2 = 8
d.backward()

print(f"a.derivative = {a.derivative}")  # d/da
print(f"b.derivative = {b.derivative}")  # d/db