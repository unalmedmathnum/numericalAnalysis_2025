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


def check_convergence_conditions(f, interval):
    """
    Verifies the fixed point convergence conditions for f(x).

    Parameters:
        f   : sympy expression of the function f(x)
        interval : tuple (a, b) where we check the conditions

    Returns:
        conditions_met (bool), explanation (str)
    """
    x = sp.symbols('x')
    f_prime = sp.diff(f, x)  # derivative f'(x)

    a, b = interval

    # Condition 1: f(x) in [a,b] for all x in [a,b]
    
    crit_points = sp.solve(f_prime, x) # Critical points: solve f'(x) = 0
    crit_points = [p for p in crit_points if p.is_real and a <= float(p.evalf()) <= b] # Keep only real critical points within [a, b]
    candidates = [f.subs(x, val) for val in [a, b] + crit_points]# Evaluate g(x) at the interval endpoints and at critical points
    # Compute minimum and maximum values of f(x) in [a, b]
    g_min = min([float(val.evalf()) for val in candidates])
    g_max = max([float(val.evalf()) for val in candidates])

    cond1 = (g_min >= a) and (g_max <= b)
    
    # Condition 2: |f'(x)| <= k < 1 for all x in (a,b)
    
    crit_points = sp.solve(sp.diff(f_prime, x), x) # Critical points: solve f''(x) = 0   
    crit_points = [p for p in crit_points if p.is_real and a <= float(p.evalf()) <= b]# Keep only real critical points within [a, b]
    candidates = [abs(f_prime.subs(x, val)) for val in [a, b] + crit_points] # Evaluate f(x) at the interval endpoints and at critical points
    k_max = max([float(c.evalf()) for c in candidates]) # Maximum value of |f'(x)| over [a, b]
    
    cond2 = k_max < 1

    # Prepare a formatted string to explain the results of the check.

    explanation = f"""
    Condition 1 (f(x) in [{a}, {b}]): {cond1}
    Condition 2 (|f'(x)| <= k < 1): {cond2}, with max |f'(x)| ≈ {k_max:.4f}
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
    
    tol = 1e-5 # Tolerance
    
    max_iter = 100 # Maximum number of iterations
    
    p0 = 1.5 # Initial guess
    
    interval = (1 , 3)  # Interval [a, b] to check convergence conditions

    
    x = sp.symbols('x') 
 
    f = sp.sqrt(x) #example fuction

    



# ==========================
  # Extra: Plotting section
    # ==========================
    g = sp.lambdify(x, f, 'numpy') #
    # Check convergence conditions
    ok, explanation = check_convergence_conditions(f, interval)
    print(explanation)

    if ok:
        root = fixed_point(g, p0, tol, max_iter)
        print("Fixed point:", root)

        # ---------- Plotting---------
        x_vals = np.linspace(0, 4, 400) # x values for plotting
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