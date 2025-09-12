from typing import Callable, TypedDict

import numpy as np


class SecantMethod(TypedDict):
    solution: int
    message: str


def secante(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    delta: float = 1e-5,
    max_iterations: int = 1000,
) -> SecantMethod:
    """
    Find a root of a function using the secant method.

    The secant method is an iterative root-finding algorithm that uses the formula:
    x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))

    This method approximates the derivative using a finite difference and doesn't
    require the analytical derivative of the function, unlike Newton's method.

    Args:
        f (Callable[[float], float]): The function for which to find the root.
                                     Must be continuous in the interval of interest.
        x0 (float): First initial guess for the root
        x1 (float): Second initial guess for the root (should be different from x0)
        delta (float): Convergence tolerance. The algorithm stops when |f(x)| < delta
        m (int): Maximum number of iterations to prevent infinite loops

    Returns:
        SecantMethod: A dictionary containing:
            - solution (float): The approximate root
            - message (str): Status message about convergence or iteration limit

    Note:
        - The method requires two initial guesses, unlike the bisection method
        - Convergence is not guaranteed and depends on the function and initial guesses
        - The method may fail if the function has discontinuities or if the
          initial guesses are poorly chosen

    Example:
        >>> f = lambda x: x**2 - 2
        >>> result = secante(f, 1, 2, 1e-7, 100)
        >>> print(f"Root: {result['solution']}")
        Root: 1.414214
    """

    if x1 == x0:
        raise ValueError("Initial guess x0 and x1 must be different")

    if delta <= 0:
        raise ValueError("Tolerance must be a positive quantity")

    if max_iterations <= 0:
        raise ValueError("Maximum number of iterations must be a positive quantity")

    for i in range(max_iterations):
        df_numeric = (f(x1) - f(x0)) / (x1 - x0)
        x = x1 - f(x1) / df_numeric

        if np.abs(f(x)) < delta:
            return {
                "solution": float(x),
                "message": f"Convergence satisfied after {i+1} iterations",
            }

        x0 = x1
        x1 = x

    return {
        "solution": float(x),
        "message": "Maximum number of iterations exceeded",
    }


# example usage of the method
if __name__ == "__main__":
    f = lambda x: x - np.exp(-x)
    x0 = 0.2
    x1 = 0.3

    delta = 1e-7
    m = 1000

    r = secante(f, x0, x1, delta, m)
    print(r)
