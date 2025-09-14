

import numpy as np
import sympy as sp

"""The Newton-Rapson method is a particular example of the fixed point method"""


def convergence_criteria(f, interval):
    """ Numerically checks the convergence criteria for the fixed-point iteration
    g(x) = x - f(x)/f'(x) over a given interval. """

    x = sp.symbols("x")
    f_prime = sp.diff(f, x)
    g = x - f / f_prime
    g_prime = sp.diff(g, x)

    # We assign g and g' into numerical functions

    g_num = sp.lambdify(x, g, "numpy")
    g_prime_num = sp.lambdify(x, g_prime, "numpy")
    f_num = sp.lambdify(x, f, "numpy")

    # lambdify takes a symbolic expression (in this case g) and converts it into a Python function
    # Then g_num(2.5) = computes g(2.5), or g_num(linspace(0,4)) computes g(x) for that array

    a, b = interval
    sample = np.linspace(a, b, 400)

    # Condition 1: g(x) on [a,b]
    vals = g_num(sample)
    vals = vals[np.isfinite(vals)]
    cond1 = len(vals) > 0 and np.all((vals >= a) & (vals <= b))

    # We evaluate g(x) along the sample and store it in vals
    # We remove infinite values or values of type nan
    # With len(vals) we make sure there is at least one data point left after filtering out problematic values
    # Finally, we make sure that all values of g(x) stored in vals lie within [a,b]

    # Condition 2: |g'(x)| < 1
    g_prime_vals = g_prime_num(sample)
    g_prime_vals = np.abs(g_prime_vals[np.isfinite(g_prime_vals)])
    k_max = np.max(g_prime_vals) if len(g_prime_vals) > 0 else np.inf
    cond2 = k_max < 1

    # We evaluate the derivative of g along the sample
    # Remove values that are infinite or nan, and take the absolute value of g'
    # Find the maximum value of |g'(x)|, but if the list is empty return infinity
    # Check the convergence condition that |g'(x)|<1

    # Condition 3: root existence
    fa, fb = f_num(a), f_num(b)
    cond3 = np.isfinite(fa) and np.isfinite(fb) and (fa * fb < 0)

    # If f(a) and f(b) are not nan or np.inf, then by the Intermediate Value Theorem
    # we check that the function crosses f(x) = 0

    explanation = f"""
    Convergence Criteria Check
    Interval: [{a}, {b}]
    1. g(x) maps the interval to itself: {cond1}
    2. |g'(x)| < 1 (Contraction mapping): {cond2}, max |g'(x)| ≈ {k_max:.4f}
    3. Root is guaranteed to exist by IVT: {cond3}

    """
    return cond1 and cond2 and cond3, explanation


def newton_raphson(f, f_prime, p0, tol, max_iter):
    """
    Finds a root of f(x) using the Newton-Raphson method.
    """
    for i in range(1, max_iter + 1):
        fp0 = f_prime(p0)
        if fp0 == 0:
            return f"Error: Derivative is zero at iteration {i}, p = {p0}"

        p = p0 - f(p0) / fp0
        try:

            if abs(p - p0) / abs(p) < tol:
                return f"Success: Converged in {i} iterations. Root ≈ {p}"

        except ZeroDivisionError:
            return f"Error: |p-p0|/|p| is |{p}-{p0}|/|{p}| at iteration {i}"

        p0 = p

    return f"Failure: Maximum iterations ({max_iter}) exceeded. The method did not converge."


if __name__ == "__main__":
    # Parameterers:
    tol = 1e-6
    max_iter = 100
    p0 = 0.5  # Initial point
    interval = (1, 5)  # Interval for theorical analysis
    x = sp.symbols('x')

    # Function
    function = x ** 3 - 2 * x + 2

    print("*** Theorical analysis of convergence ***")
    converges_guaranteed, msg = convergence_criteria(function, interval)
    print(msg)

    if not converges_guaranteed:
        print("Warning: Convergence is not guaranteed in the interval for any initial point.\n"
              "The success of the method will depend on the proximity of the point p0 to the root.\n")

    print("*** Newton-Raphson execution ***")
    f_numeric = sp.lambdify(x, function, 'numpy')
    f_prime_numeric = sp.lambdify(x, sp.diff(function, x), 'numpy')

    result = newton_raphson(f_numeric, f_prime_numeric, p0, tol, max_iter)
    print(f"Initial point p0 = {p0}")
    print(f"Result: {result}")





