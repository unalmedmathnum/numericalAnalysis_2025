from typing import Callable, TypedDict

import numpy as np


class SecantMethod(TypedDict):
    solution: int
    message: str


def secante(
    f: Callable[[float], float], x0: float, x1: float, delta: float, m: int
) -> SecantMethod:
    for i in range(m):
        x = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))

        if np.abs(f(x)) < delta:
            return {
                "solution": x,
                "message": f"Convergence satisfied after {i+1} iterations",
            }

        x0 = x1
        x1 = x

    return {
        "solution": x,
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
