import numpy as np


def newton_modified(f, df, d2f, x0, delta, m):
    x = x0

    for i in range(m):
        x = x - f(x) * df(x) / (df(x) ** 2 - f(x) * d2f(x))

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

    r = newton_modified(f, df, d2f, x0, delta, m)
    print(r)
