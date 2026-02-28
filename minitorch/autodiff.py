"""
Automatic differentiation utilities for MiniTorch.
"""

from typing import Callable, List, Tuple, Any


def central_difference(f: Callable[..., float], *vals: float, arg: int = 0, epsilon: float = 1e-6) -> float:
    vals_list = list(vals)

    vals_plus = vals_list.copy()
    vals_plus[arg] = vals_plus[arg] + epsilon

    vals_minus = vals_list.copy()
    vals_minus[arg] = vals_minus[arg] - epsilon

    f_plus = f(*vals_plus)
    f_minus = f(*vals_minus)

    return (f_plus - f_minus) / (2 * epsilon)

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class Variable:
    """
    A node in the computation graph.

    Attributes:
        history: Record of the operation that created this variable
        derivative: Accumulated gradient (set during backward pass)
        name: Optional name for debugging
    """
    history: Optional["History"] = None
    derivative: Optional[float] = None
    name: Optional[str] = None

    def is_leaf(self) -> bool:
        """A leaf variable has no history (was not created by an operation)."""
        return self.history is None

    def is_constant(self) -> bool:
        """A constant has no history and will not receive gradients."""
        return self.history is None

    def requires_grad_(self, requires_grad: bool = True) -> "Variable":
        """Set whether this variable should track gradients."""
        if requires_grad:
            self.history = History()
        else:
            self.history = None
        return self


@dataclass
class History:
    """
    Records the operation that created a variable.

    Attributes:
        last_fn: The function class that created this variable
        ctx: Context object storing values needed for backward
        inputs: The input variables to the operation
    """
    last_fn: Optional[type] = None
    ctx: Optional["Context"] = None
    inputs: Sequence["Variable"] = ()

def backpropagate(final_var: Variable, deriv: float = 1.0) -> None:
    """
    Run reverse-mode autodiff starting from final_var.
    """

    stack = [(final_var, deriv)]

    while stack:
        var, d = stack.pop()

        # accumulate gradient
        if hasattr(var, "accumulate_derivative"):
            var.accumulate_derivative(d)

        # stop if leaf
        if var.history is None or var.history.last_fn is None:
            continue

        h = var.history

        grads = h.last_fn.backward(h.ctx, d)

        for inp, g in zip(h.inputs, grads):
            stack.append((inp, g))

def topological_sort(variable: Variable) -> List[Variable]:
    order: List[Variable] = []
    visited: Set[int] = set()

    def visit(var: Variable) -> None:
        var_id = id(var)
        if var_id in visited:
            return
        visited.add(var_id)

        if var.history is not None and var.history.inputs:
            for input_var in var.history.inputs:
                visit(input_var)

        order.append(var)

    visit(variable)
    return order