from typing import Callable

import numpy as np
import pandas as pd


def bisection(
    f: Callable, a: float, b: float, delta: float
) -> tuple[float, pd.DataFrame]:
    """
    Find the root of a continuous function using the bisection method.

    The bisection method is a root-finding algorithm that repeatedly bisects an interval
    and selects the subinterval where the function changes sign. This guarantees
    convergence to a root if one exists in the given interval.

    Parameters:
    -----------
    f : Callable
        A continuous function for which we want to find a root (i.e., f(x) = 0).
        The function should take a single float argument and return a float.

    a : float
        Left endpoint of the initial interval. Must satisfy f(a) * f(b) < 0.

    b : float
        Right endpoint of the initial interval. Must satisfy f(a) * f(b) < 0.

    delta : float
        Desired precision/tolerance for the root approximation.
        The algorithm will continue until the interval width is less than delta.
        Must be positive.

    Returns:
    --------
    tuple[float, pd.DataFrame]
        A tuple containing:
        - float: The approximate root of the function
        - pd.DataFrame: A DataFrame with columns ['ak', 'bk', 'pk'] showing
          the iteration history, where:
          * ak: left endpoint at iteration k
          * bk: right endpoint at iteration k
          * pk: midpoint at iteration k

    Raises:
    -------
    ValueError
        If f(a) * f(b) > 0, meaning the Intermediate Value Theorem is not satisfied
        and there's no guarantee of a root in the interval [a, b].

    Algorithm:
    ----------
    1. Verify that f(a) and f(b) have opposite signs
    2. Calculate the maximum number of iterations needed: k = ceil(log2((b-a)/delta))
    3. For each iteration:
       - Calculate midpoint pk = (ak + bk) / 2
       - If f(ak) * f(pk) < 0: root is in [ak, pk], so set bk = pk
       - If f(ak) * f(pk) > 0: root is in [pk, bk], so set ak = pk
       - If f(ak) * f(pk) = 0: pk is exactly a root, return immediately
    4. Return the final midpoint approximation

    Notes:
    ------
    - The function assumes f is continuous on [a, b]
    - Convergence is guaranteed if f(a) * f(b) < 0
    - The method has linear convergence with rate 1/2
    - Time complexity: O(log((b-a)/delta))
    - The actual precision achieved may be better than delta

    Example:
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from typing import Callable
    >>>
    >>> # Find root of f(x) = x^2 - 2 (should be approximately sqrt(2))
    >>> def f(x):
    ...     return x**2 - 2
    >>>
    >>> root, iterations = bisection(f, 0, 2, 0.001)
    >>> print(f"Root: {root:.6f}")
    >>> print(f"Verification: f({root:.6f}) = {f(root):.6f}")
    >>> print(f"Number of iterations: {len(iterations)}")
    >>> print(iterations.head())
    """

    if f(a) * f(b) > 0:
        raise ValueError(
            "Intermediate Value Theorem is not satisfied: f(a) and f(b) must have opposite signs. "
            f"Got f({a}) = {f(a)} and f({b}) = {f(b)}"
        )

    k = int(np.ceil(np.log2((b - a) / delta)))
    ak = a
    bk = b
    L = []
    for _ in range(k):
        pk = (ak + bk) / 2
        L.append((ak, bk, pk))
        if f(ak) * f(pk) < 0:
            bk = pk
        elif f(ak) * f(pk) > 0:
            ak = pk
        else:
            return pk, pd.DataFrame(L, columns=["ak", "bk", "pk"])
    return pk, pd.DataFrame(L, columns=["ak", "bk", "pk"])


# Example of bisection implementation
if __name__ == "__main__":
    f = lambda x: x - np.exp(-x)
    a = 0
    b = 1
    delta = 1e-6
    p, df = bisection(f, a, b, delta)

    print(p)
    print(df)
