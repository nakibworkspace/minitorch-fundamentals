#!/bin/bash

# MiniTorch Module 0 Project Setup Script
# This script creates the complete project structure for students

set -e

echo "Setting up MiniTorch Module 0 project structure..."

# Create directory structure
mkdir -p minitorch tests project/interface

echo "Creating project structure..."

# Create main package files
cat > minitorch/__init__.py << 'EOF'
"""
MiniTorch: A minimal deep learning library for educational purposes.

Module 0: ML Programming Foundations
"""

__version__ = "0.1.0"

# We'll import our implementations as we build them
# from .operators import *  # Module 0.1
# from .module import *     # Module 0.4
EOF

# Create operators.py skeleton
cat > minitorch/operators.py << 'EOF'
"""
Mathematical operators for MiniTorch.

These form the foundation of all neural network operations.
You'll implement each function to understand how deep learning
frameworks handle basic mathematics.
"""

import math
from typing import Callable, Iterable


# TODO: Implement these functions in Task 0.1
def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    raise NotImplementedError("Implement in Task 0.1")


def id(x: float) -> float:
    """Identity function."""
    raise NotImplementedError("Implement in Task 0.1")


def add(x: float, y: float) -> float:
    """Add two numbers."""
    raise NotImplementedError("Implement in Task 0.1")


def neg(x: float) -> float:
    """Negate a number."""
    raise NotImplementedError("Implement in Task 0.1")


def lt(x: float, y: float) -> float:
    """Less than comparison."""
    raise NotImplementedError("Implement in Task 0.1")


def eq(x: float, y: float) -> float:
    """Equality comparison."""
    raise NotImplementedError("Implement in Task 0.1")


def max(x: float, y: float) -> float:
    """Maximum of two numbers."""
    raise NotImplementedError("Implement in Task 0.1")


def is_close(x: float, y: float) -> float:
    """Check if numbers are close."""
    raise NotImplementedError("Implement in Task 0.1")


def sigmoid(x: float) -> float:
    """Sigmoid activation function."""
    raise NotImplementedError("Implement in Task 0.1")


def relu(x: float) -> float:
    """ReLU activation function."""
    raise NotImplementedError("Implement in Task 0.1")


def log(x: float) -> float:
    """Natural logarithm."""
    raise NotImplementedError("Implement in Task 0.1")


def exp(x: float) -> float:
    """Exponential function."""
    raise NotImplementedError("Implement in Task 0.1")


def inv(x: float) -> float:
    """Reciprocal function."""
    raise NotImplementedError("Implement in Task 0.1")


def log_back(x: float, grad: float) -> float:
    """Gradient of log."""
    raise NotImplementedError("Implement in Task 0.1")


def inv_back(x: float, grad: float) -> float:
    """Gradient of inv."""
    raise NotImplementedError("Implement in Task 0.1")


def relu_back(x: float, grad: float) -> float:
    """Gradient of ReLU."""
    raise NotImplementedError("Implement in Task 0.1")


# TODO: Implement these in Task 0.3
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order map function."""
    raise NotImplementedError("Implement in Task 0.3")


def zipWith(fn: Callable[[float, float], float]) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipWith function."""
    raise NotImplementedError("Implement in Task 0.3")


def reduce(fn: Callable[[float, float], float], init: float) -> Callable[[Iterable[float]], float]:
    """Higher-order reduce function."""
    raise NotImplementedError("Implement in Task 0.3")


def sum(ls: Iterable[float]) -> float:
    """Sum using reduce."""
    raise NotImplementedError("Implement in Task 0.3")


def prod(ls: Iterable[float]) -> float:
    """Product using reduce."""
    raise NotImplementedError("Implement in Task 0.3")


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate list using map."""
    raise NotImplementedError("Implement in Task 0.3")


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add lists using zipWith."""
    raise NotImplementedError("Implement in Task 0.3")
EOF

# Create module.py skeleton
cat > minitorch/module.py << 'EOF'
"""
Module system for organizing neural network components and parameters.

This provides the foundation for building complex neural networks
with proper parameter management and hierarchy.
"""

from typing import Dict, List, Tuple, Any, Sequence


class Parameter:
    """A trainable parameter in a neural network."""
    
    def __init__(self, value: Any):
        self.value = value
        
    @property
    def shape(self):
        """Return parameter shape."""
        if hasattr(self.value, 'shape'):
            return self.value.shape
        return ()
    
    def __repr__(self):
        return f"Parameter(value={self.value})"


class Module:
    """Base class for all neural network modules."""
    
    def __init__(self):
        # TODO: Initialize in Task 0.4
        raise NotImplementedError("Implement in Task 0.4")
    
    def modules(self) -> Sequence["Module"]:
        """Return all sub-modules."""
        raise NotImplementedError("Implement in Task 0.4")
    
    def train(self):
        """Set training mode."""
        raise NotImplementedError("Implement in Task 0.4")
    
    def eval(self):
        """Set evaluation mode."""
        raise NotImplementedError("Implement in Task 0.4")
    
    def named_parameters(self) -> Sequence[Tuple[str, Parameter]]:
        """Return all parameters with names."""
        raise NotImplementedError("Implement in Task 0.4")
    
    def parameters(self) -> Sequence[Parameter]:
        """Return all parameters."""
        raise NotImplementedError("Implement in Task 0.4")
    
    def add_parameter(self, name: str, value: Any) -> Parameter:
        """Add a parameter."""
        raise NotImplementedError("Implement in Task 0.4")
    
    def __setattr__(self, key: str, value: Any):
        """Custom attribute setter."""
        raise NotImplementedError("Implement in Task 0.4")
EOF

# Create testing utilities
cat > minitorch/testing.py << 'EOF'
"""Testing utilities for MiniTorch."""

def assert_close(a: float, b: float, eps: float = 1e-2):
    """Assert two floats are close within tolerance."""
    assert abs(a - b) < eps, f"Values not close: {a} vs {b}"
EOF

# Create datasets.py skeleton
cat > minitorch/datasets.py << 'EOF'
"""
Dataset generators for classification experiments.

You'll implement documentation for these functions in Task 0.5.
"""

import random
from typing import List, Tuple


def simple(N: int) -> Tuple[List[Tuple[float, float]], List[int]]:
    """
    Simple linear dataset - points on left vs right.
    
    TODO: Add complete docstring in Task 0.5
    
    Classification rule: x_coord >= 0.5 -> class 1, else class 0
    """
    points = []
    labels = []
    for _ in range(N):
        x = random.random()
        y = random.random()
        points.append((x, y))
        labels.append(1 if x >= 0.5 else 0)
    return points, labels


def diag(N: int) -> Tuple[List[Tuple[float, float]], List[int]]:
    """
    Diagonal dataset - points above vs below main diagonal.
    
    TODO: Add complete docstring in Task 0.5
    
    Classification rule: x + y >= 1.0 -> class 1, else class 0
    """
    points = []
    labels = []
    for _ in range(N):
        x = random.random()
        y = random.random()
        points.append((x, y))
        labels.append(1 if x + y >= 1.0 else 0)
    return points, labels


def split(N: int) -> Tuple[List[Tuple[float, float]], List[int]]:
    """
    Split dataset - points in center vs edges.
    
    TODO: Add complete docstring in Task 0.5
    
    Classification rule: 0.2 <= x <= 0.8 -> class 0, else class 1
    """
    points = []
    labels = []
    for _ in range(N):
        x = random.random()
        y = random.random()
        points.append((x, y))
        labels.append(0 if 0.2 <= x <= 0.8 else 1)
    return points, labels


def xor(N: int) -> Tuple[List[Tuple[float, float]], List[int]]:
    """
    XOR dataset - requires non-linear separation.
    
    TODO: Add complete docstring in Task 0.5
    
    Classification rule:
        (x < 0.5 and y < 0.5) or (x >= 0.5 and y >= 0.5) -> class 0
        else -> class 1
    """
    points = []
    labels = []
    for _ in range(N):
        x = random.random()
        y = random.random()
        points.append((x, y))
        same_side = (x < 0.5) == (y < 0.5)
        labels.append(0 if same_side else 1)
    return points, labels
EOF

echo "Creating test suite..."

# Create test_operators.py
cat > tests/test_operators.py << 'EOF'
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
def test_sigmoid_properties(a):
    """Test mathematical properties of sigmoid function."""
    if math.isfinite(a):
        sig_a = sigmoid(a)
        # Property 1: Output bounded between 0 and 1
        assert 0 < sig_a < 1
        # Property 2: sigmoid(0) = 0.5
        if is_close(a, 0.0) == 1.0:
            assert is_close(sig_a, 0.5) == 1.0
        # Property 3: sigmoid(-x) = 1 - sigmoid(x)
        sig_neg_a = sigmoid(-a)
        expected = 1.0 - sig_a
        assert is_close(sig_neg_a, expected) == 1.0

@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_transitive(a, b, c):
    """Test transitive property: if a < b and b < c, then a < c."""
    if all(math.isfinite(v) for v in [a, b, c]):
        if lt(a, b) == 1.0 and lt(b, c) == 1.0:
            assert lt(a, c) == 1.0

@pytest.mark.task0_2
@given(small_floats, small_floats)
def test_symmetric(x, y):
    """Test that multiplication is commutative."""
    if math.isfinite(x) and math.isfinite(y):
        assert_close(mul(x, y), mul(y, x))

@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_distribute(x, y, z):
    """Test distributive property: z * (x + y) = z*x + z*y."""
    if all(math.isfinite(v) for v in [x, y, z]):
        left_side = mul(z, add(x, y))
        right_side = add(mul(z, x), mul(z, y))
        assert_close(left_side, right_side)

@pytest.mark.task0_2
@given(small_floats)
def test_other(a):
    """Test additive inverse property: a + (-a) = 0."""
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
EOF

# Create test_module.py  
cat > tests/test_module.py << 'EOF'
"""Tests for module system."""

import pytest
from minitorch.module import Module, Parameter

@pytest.mark.task0_4
def test_module_init():
    module = Module()
    assert module.training == True

@pytest.mark.task0_4
def test_train_eval():
    module = Module()
    assert module.training == True
    
    module.eval()
    assert module.training == False
    
    module.train()
    assert module.training == True

@pytest.mark.task0_4
def test_parameter():
    param = Parameter(5.0)
    assert param.value == 5.0

@pytest.mark.task0_4
def test_add_parameter():
    module = Module()
    param = module.add_parameter("weight", 3.14)
    assert isinstance(param, Parameter)
    assert param.value == 3.14

@pytest.mark.task0_4
def test_nested_modules():
    parent = Module()
    child = Module()
    parent.child = child
    
    modules = list(parent.modules())
    assert child in modules
EOF

# Create pytest configuration
cat > tests/conftest.py << 'EOF'
"""Pytest configuration for MiniTorch."""

import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "task0_1: Task 0.1 operators")
    config.addinivalue_line("markers", "task0_2: Task 0.2 property tests") 
    config.addinivalue_line("markers", "task0_3: Task 0.3 functional")
    config.addinivalue_line("markers", "task0_4: Task 0.4 modules")
EOF

echo "Creating package files..."

# Create requirements.txt
cat > requirements.txt << 'EOF'
pytest>=6.0
hypothesis>=6.0
streamlit>=1.12.0
matplotlib>=3.5.0
numpy>=1.21.0
black>=22.0
flake8>=4.0
EOF

# Create setup.py for package installation
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="minitorch",
    version="0.1.0",
    description="A minimal deep learning library for educational purposes",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pytest>=6.0",
        "hypothesis>=6.0",
    ],
    extras_require={
        "viz": ["streamlit>=1.12.0", "matplotlib>=3.5.0", "numpy>=1.21.0"],
        "dev": ["black>=22.0", "flake8>=4.0"],
    },
)
EOF

echo "Creating visualization app..."

# Create Streamlit demo app
cat > project/app.py << 'EOF'
"""
Streamlit app for visualizing MiniTorch operations.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("MiniTorch Module 0 - Mathematical Operators")

st.sidebar.header("Function Visualizer")

# Import your operators (once implemented)
try:
    from minitorch.operators import sigmoid, relu, add, mul
    operators_available = True
except (ImportError, NotImplementedError):
    st.error("Operators not yet implemented. Complete Task 0.1 first!")
    operators_available = False

if operators_available:
    func = st.sidebar.selectbox("Select function to visualize:", 
                               ["sigmoid", "relu"])
    
    x = np.linspace(-5, 5, 100)
    
    try:
        if func == "sigmoid":
            y = [sigmoid(xi) for xi in x]
            st.write("## Sigmoid Function")
            st.write("Formula: σ(x) = 1 / (1 + e^(-x))")
            
        elif func == "relu":
            y = [relu(xi) for xi in x] 
            st.write("## ReLU Function")
            st.write("Formula: ReLU(x) = max(0, x)")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, linewidth=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel(f'{func}(x)', fontsize=12)
        ax.set_title(f'{func.title()} Function', fontsize=14)
        st.pyplot(fig)
        
    except NotImplementedError:
        st.error(f"{func} function not yet implemented!")

st.write("## Test Your Operators")

if operators_available:
    col1, col2 = st.columns(2)
    
    with col1:
        a = st.number_input("Enter first number:", value=2.0)
        b = st.number_input("Enter second number:", value=3.0)
        
    with col2:
        try:
            if st.button("Test Add"):
                result = add(a, b)
                st.success(f"add({a}, {b}) = {result}")
        except NotImplementedError:
            st.error("Add function not implemented!")
            
        try:
            if st.button("Test Multiply"):  
                result = mul(a, b)
                st.success(f"mul({a}, {b}) = {result}")
        except NotImplementedError:
            st.error("Multiply function not implemented!")
            
        try:
            if st.button("Test Sigmoid"):
                result = sigmoid(a)
                st.success(f"sigmoid({a}) = {result:.4f}")
        except NotImplementedError:
            st.error("Sigmoid function not implemented!")
            
        try:
            if st.button("Test ReLU"):
                result = relu(a)
                st.success(f"relu({a}) = {result}")
        except NotImplementedError:
            st.error("ReLU function not implemented!")

st.write("---")
st.write("### Testing Progress")

# Show testing progress
try:
    import subprocess
    result = subprocess.run(['python', '-m', 'pytest', '--tb=no', '-q'], 
                          capture_output=True, text=True, cwd='.')
    if result.returncode == 0:
        st.success("All tests passing!")
    else:
        st.warning("Some tests still failing. Keep implementing!")
except:
    st.info("Run `pytest` in terminal to check your progress")

st.write("### Tasks Completed")
st.write("- [ ] Task 0.1: Mathematical Operators")  
st.write("- [ ] Task 0.2: Property Testing")
st.write("- [ ] Task 0.3: Functional Programming")
st.write("- [ ] Task 0.4: Module System")
EOF

# Create project README
cat > README.md << 'EOF'
# MiniTorch Module 0: ML Programming Foundations

Welcome to MiniTorch! You'll build a deep learning library from scratch to understand how frameworks like PyTorch work internally.

## Project Structure

```
minitorch/
├── minitorch/           # Main package
│   ├── operators.py     # Mathematical operations (Task 0.1)
│   ├── module.py        # Parameter management (Task 0.4)  
│   └── testing.py       # Utility functions
├── tests/               # Test suite
├── project/             # Demo applications
└── requirements.txt     # Dependencies
```

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Run tests (will fail initially):**
   ```bash
   pytest -m task0_1 -v
   ```

3. **Launch visualization:**
   ```bash
   streamlit run project/app.py
   ```

4. **Implement functions in:**
   - `minitorch/operators.py` (Tasks 0.1 & 0.3)
   - `minitorch/module.py` (Task 0.4)

## Tasks

- **Task 0.1**: Mathematical operators (add, mul, sigmoid, relu, etc.)
- **Task 0.2**: Property-based testing (already implemented)
- **Task 0.3**: Functional programming (map, reduce, zipWith)
- **Task 0.4**: Module system for parameters

## Testing

```bash
# Test specific task
pytest -m task0_1 -xvs

# Test all tasks
pytest -m "task0_1 or task0_2 or task0_3 or task0_4" -v

# Check code style
black minitorch/ tests/
flake8 minitorch/ tests/
```

## Success Criteria

- All tests passing
- Streamlit app works
- Ready for Module 1!

---

*Good luck building your deep learning library!*
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Testing
.coverage
.pytest_cache/
htmlcov/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Virtual environment
venv/
env/
ENV/

# OS
.DS_Store
Thumbs.db
EOF

echo "Project structure created successfully!"
echo ""
echo "Files created:"
echo "   - minitorch/__init__.py"
echo "   - minitorch/operators.py (skeleton with TODOs)"
echo "   - minitorch/module.py (skeleton with TODOs)"
echo "   - minitorch/testing.py"
echo "   - minitorch/datasets.py (dataset generators)"
echo "   - tests/test_operators.py"
echo "   - tests/test_module.py"
echo "   - tests/conftest.py"
echo "   - project/app.py (Streamlit visualization)"
echo "   - requirements.txt"
echo "   - setup.py"
echo "   - README.md"
echo "   - .gitignore"
echo ""
echo "Next steps:"
echo "   1. pip install -r requirements.txt"
echo "   2. pip install -e ."
echo "   3. pytest -m task0_1 -v  (should fail - that's expected!)"
echo "   4. code minitorch/operators.py  (start implementing!)"
echo "   5. streamlit run project/app.py  (visualize your work)"
echo ""
echo "Goal: Make all tests pass by implementing the TODO functions!"
echo "Tip: Start with Task 0.1 in minitorch/operators.py"
