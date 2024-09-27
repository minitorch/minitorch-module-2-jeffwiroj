from dataclasses import dataclass
import enum
from signal import valid_signals
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    x1 = []
    x2 = []
    for i, val in enumerate(vals):
        if i == arg:
            x1.append(val + epsilon)
            x2.append(val - epsilon)
        else:
            x1.append(val)
            x2.append(val)
    return (f(*x1) - f(*x2)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visited = set()
    path = []

    def dfs(scalar):
        if scalar.is_constant():
            return
        if scalar.unique_id in visited:
            return
        for ngh in scalar.history.inputs:
            dfs(ngh)
        visited.add(scalar.unique_id)
        path.append(scalar)

    dfs(variable)
    return path[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    path = topological_sort(variable)
    scalar_deriv = {}
    for i, scalar in enumerate(path):
        if i == 0:
            scalar_deriv[scalar.unique_id] = deriv
        else:
            scalar_deriv[scalar.unique_id] = 0

    for scalar in path:
        dout = scalar_deriv[scalar.unique_id]
        if scalar.is_leaf():
            scalar.accumulate_derivative(dout)
        else:
            scalar_grad = scalar.chain_rule(dout)
            for parent, grad in scalar_grad:
                if parent.is_constant():
                    continue
                scalar_deriv[parent.unique_id] += grad


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
