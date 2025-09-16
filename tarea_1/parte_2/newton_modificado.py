from typing import Callable, TypedDict

import numpy as np

class NewtonModified(TypedDict):
    solution: float
    message: str


def newton_modified(
    f: Callable[[float], float],
    df: Callable[[float], float],
    d2f: Callable[[float], float],
    x0: float,
    delta: float,
    m: int,
) -> NewtonModified:
    """
    Parameters
    ----------
    f : callable
        The function for which to find the root. Should accept a scalar and return a scalar.
    df : callable
        The first derivative of function f. Should accept a scalar and return a scalar.
    d2f : callable
        The second derivative of function f. Should accept a scalar and return a scalar.
    x0 : float
        Initial guess for the root.
    delta : float
        Convergence tolerance. The algorithm stops when |f(x)| < delta.
    m : int
        Maximum number of iterations allowed.

    Returns
    -------
    dict
        A dictionary containing:
        - 'solution': float - The approximate root found
        - 'message': str - Status message indicating convergence or maximum iterations reached

    Notes
    -----
    The iteration formula used is:
    x_{n+1} = x_n - f(x_n)*f'(x_n) / ([f'(x_n)]^2 - f(x_n)*f''(x_n))

    This method may fail if:
    - The denominator becomes zero or very small
    - The derivatives don't exist or are discontinuous
    - The initial guess is too far from the root
    - The function has multiple roots in the search region
    """
    x = x0

    for i in range(m):
        fx = f(x)
        dfx = df(x)
        d2fx = d2f(x)

        denominator = dfx**2 - fx * d2fx

        if abs(denominator) < 1e-15:
            return {
                "solution": float(x),
                "message": f"Method failed: denominator too small at iteration {i+1}",
            }

        x = x - fx * dfx / denominator

        if np.abs(f(x)) < delta:
            return {
                "solution": float(x),
                "message": f"Convergence satisfied after {i+1} iterations",
            }

    return {
        "solution": float(x),
        "message": "Maximum number of iterations exceeded",
    }


if __name__ == "__main__":
    f = lambda x: np.exp(-x) - x
    df = lambda x: -np.exp(-x) - 1
    d2f = lambda x: np.exp(-x)

    x0 = 0.001
    delta = 1e-5
    m = 1000

    result = newton_modified(f, df, d2f, x0, delta, m)
