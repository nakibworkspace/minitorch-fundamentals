"""Tests for operators module."""

import pytest
from hypothesis import given
from hypothesis.strategies import floats
import math

from minitorch.operators import *
from minitorch.testing import assert_close

# Task 0.1 Tests
@pytest.mark.task0_1
def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

@pytest.mark.task0_1  
def test_mul():
    assert mul(3, 4) == 12
    assert mul(-2, 3) == -6

@pytest.mark.task0_1
def test_sigmoid():
    assert_close(sigmoid(0), 0.5)
    assert 0 < sigmoid(-10) < 1
    assert 0 < sigmoid(10) < 1

@pytest.mark.task0_1
def test_relu():
    assert relu(5) == 5
    assert relu(-3) == 0
    assert relu(0) == 0

# Task 0.2 Tests (Property-based)
# Strategy for floats in a reasonable range
small_floats = floats(min_value=-100, max_value=100)

@pytest.mark.task0_2
@given(small_floats)
def test_sigmoid(a):
    from minitorch.operators import sigmoid, is_close

    if math.isfinite(a):
        sig_a = sigmoid(a)
        assert 0 < sig_a < 1
        if is_close(a, 0.0) == 1.0:
            assert is_close(sig_a, 0.5) == 1.0
        sig_neg_a = sigmoid(-a)
        expected = 1.0 - sig_a
        assert is_close(sig_neg_a, expected) == 1.0


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_transitive(a, b, c):
    from minitorch.operators import lt

    if all(math.isfinite(v) for v in [a, b, c]):
        if lt(a, b) == 1.0 and lt(b, c) == 1.0:
            assert lt(a, c) == 1.0


@pytest.mark.task0_2
@given(small_floats, small_floats)
def test_symmetric(x, y):
    from minitorch.operators import mul
    from minitorch.testing import assert_close

    if math.isfinite(x) and math.isfinite(y):
        assert_close(mul(x, y), mul(y, x))


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_distribute(x, y, z):
    from minitorch.operators import mul, add
    from minitorch.testing import assert_close

    if all(math.isfinite(v) for v in [x, y, z]):
        left_side = mul(z, add(x, y))
        right_side = add(mul(z, x), mul(z, y))
        assert_close(left_side, right_side)


@pytest.mark.task0_2
@given(small_floats)
def test_other(a):
    from minitorch.operators import add, neg
    from minitorch.testing import assert_close

    if math.isfinite(a):
        result = add(a, neg(a))
        assert_close(result, 0.0)

# Task 0.3 Tests
@pytest.mark.task0_3
def test_map():
    negate = map(neg)
    assert list(negate([1, 2, 3])) == [-1, -2, -3]

@pytest.mark.task0_3
def test_zipWith():
    add_lists = zipWith(add)
    assert list(add_lists([1, 2], [3, 4])) == [4, 6]

@pytest.mark.task0_3
def test_reduce():
    sum_fn = reduce(add, 0)
    assert sum_fn([1, 2, 3, 4]) == 10
    
    prod_fn = reduce(mul, 1)
    assert prod_fn([2, 3, 4]) == 24

@pytest.mark.task0_3
def test_sum():
    assert sum([1, 2, 3, 4]) == 10
    assert sum([]) == 0

@pytest.mark.task0_3
def test_prod():
    assert prod([2, 3, 4]) == 24
    assert prod([]) == 1
