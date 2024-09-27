"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, Optional

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# 0.1
def mul(a: float, b: float):
    return a * b


def id(a: float):
    return a


def add(a: float, b: float):
    return a + b


def neg(a: float):
    return -1.0 * a


def lt(a: float, b: float):
    return 1.0 if a < b else 0.0


def eq(a: float, b: float):
    return 1.0 if a == b else 0.0


def max(a: float, b: float):
    return a if a - b > 0 else b


def sigmoid(a: float):
    return 1.0 / (1 + math.exp(-a))


def relu(a: float):
    return float(a) if a > 0 else 0.0


def log(a: float):
    return float(math.log(a))


def exp(a: float):
    return float(math.exp(a))


def inv(a: float):
    return 1.0 / a


def log_back(a: float, b: float):
    return inv(a) * b


def inv_back(a: float, b: float):
    return (-1.0 / (a**2)) * b


def relu_back(a: float, b: float):
    return 0.0 if a <= 0 else b


def is_close(a: float, b: float):
    return 1.0 if (a - b) < 1e-7 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# Implement for Task 0.3.
def map(f: Callable, iterable: Iterable):
    return [f(elem) for elem in iterable]


def zipWith(f: Callable, l1: Iterable, l2: Iterable):
    return [f(x1, x2) for x1, x2 in zip(l1, l2)]


def addLists(l1: Iterable, l2: Iterable):
    return zipWith(lambda x, y: x + y, l1, l2)


def negList(l: Iterable):
    return map(lambda x: neg(x), l)


def reduce(f: Callable, l: Iterable, val: Optional[float] = None) -> float:

    for item in l:
        if val is None:
            val = item
        else:
            val = f(val, item)
    return val if val is not None else -1


def sum(l: Iterable) -> float:
    return reduce(lambda x, y: x + y, l)


def prod(l: Iterable) -> float:
    return reduce(lambda x, y: x * y, l)
