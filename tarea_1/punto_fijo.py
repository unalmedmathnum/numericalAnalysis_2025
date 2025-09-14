"""fixed_point"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

"""

A fixed point p is one such that p = f(p). This root-finding method works by applying 
the Fixed Point Theorem.

Verifies the Fixed Point Theorem's convergence conditions for a function f(x).

    The theorem guarantees a unique fixed point in [a, b] if:
    1. f(x) is continuous on [a, b] and maps [a, b] into [a, b].
    2. The derivative |f'(x)| <= k < 1 for all x in (a, b) (i.e., g is a contraction).


Parameters:
  - f = the function we're evaluating (function)
  - p0 = the starting point (float)
  - tol = the maximum error tolerance allowed (float)
  - max_iter = the maximum number of iterations allowed (int)
  - interval : tuple (a, b) where we check the conditions

Returns:
  - tuple: A boolean indicating if conditions are met, and a string with a detailed explanation.
  - p_new = the approximated value of the fixed point (float)
  - If it isn't able to find one, it returns an error message (string)
  - Generates and displays a plot using Matplotlib. This graph visualizes the function g(x), the 
    reference line y = x and the approximated value of the fixed point.
"""


def check_convergence_conditions(f_expr, interval):
    """
    Verifies the fixed point convergence conditions for f(x).

    Parameters:
        f   : sympy expression of the function f(x)
        interval : tuple (a, b) where we check the conditions

    Returns:
        conditions_met (bool), explanation (str)
    """
    x = sp.symbols('x')
    g = f
    g_prime = sp.diff(g, x)  # derivative g'(x)

    a, b = interval

    # Condition 1: g(x) in [a,b] for all x in [a,b]
    # Check range roughly by evaluating g(x) at points in the interval
    vals = [g.subs(x, val) for val in np.linspace(a, b, 10000)]
    cond1 = all(a <= val <= b for val in vals)

    # Condition 2: |g'(x)| <= k < 1 for all x in (a,b)
    g_prime_vals = [abs(g_prime.subs(x, val)) for val in np.linspace(a, b, 10000)]
    k_max = max(g_prime_vals)
    cond2 = k_max < 1

    # Prepare a formatted string to explain the results of the check.

    explanation = f"""
    Condition 1 (g(x) in [{a}, {b}]): {cond1}
    Condition 2 (|g'(x)| <= k < 1): {cond2}, with max |g'(x)| ≈ {k_max:.4f}
    """
    return cond1 and cond2, explanation


def fixed_point(f, p0, tol, max_iter):
    p = p0
    for i in range(max_iter):
        p_new = f(p)
        if np.abs(p_new - p) < tol:
            print(f"Converged in {i} iterations.")
            return p_new
        p = p_new
    return "Error: Iterations exceeded. The method did not converge"

#-----------------------Example -----------------------------------
if __name__ == "__main__":
    
    tol = 1e-4
    
    max_iter = 100
    
    p0 = 0.5
    
    interval = (0 , 4)

    # Define g(x) in sympy
    x = sp.symbols('x')

    f = x/2 + 1 #example fuction

    g = sp.lambdify(x, f, 'numpy')



# ==========================
  # Extra: Plotting section
    # ==========================
    # Check convergence conditions
    ok, explanation = check_convergence_conditions(f, interval)
    print(explanation)

    if ok:
        root = fixed_point(g, p0, tol, max_iter)
        print("Fixed point:", root)

        # ---------- Plotting---------
        x_vals = np.linspace(0, 4, 400)
        y_vals = g(x_vals)

        plt.figure(figsize=(6, 5))
        plt.plot(x_vals, y_vals, label="f(x)")
        plt.plot(x_vals, x_vals, 'k--', label="y = x")
        if isinstance(root, (int, float)): # Check if root is a number
            plt.scatter(root, g(root), color="red", zorder=5, label=f"Fixed point ≈ {root:.4f}")

        # Plot configuration
        plt.title("Fixed Point Method with Convergence Check")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Convergence conditions are NOT satisfied in the given interval.")