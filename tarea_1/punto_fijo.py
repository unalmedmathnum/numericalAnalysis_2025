"""fixed_point"""

import numpy as np
import matplotlib.pyplot as plt

"""
This root search method works by using the 'fixed point theorem', that states that
if f is contractive (that is, it's derivative is lesser than 1), then it has a fixed point
A fixed point p is one such that p = f(p). We'll use this to our advantage to find the roots
of the function

Parameters:
  - f = the function we're evaluating (function)
  - p0 = the starting point (float)
  - tol = the maximum error tolerance allowed (float)
  - max_iter = the maximum number of iterations allowed (int)

Returns:
  - p_new = the approximated value of the fixed point (float)
  - If it isn't able to find one, it returns an error message (string)
"""

def fixed_point(f, p0, tol, max_iter):
  p = p0 #We fix the starting point as p
  for i in range(max_iter):
    p_new = f(p)
    if np.abs(p_new - p) < tol: #If the tolerance is greater than the error, then it returns it as the value
      print(f"Converged in {i} iterations.")
      return p_new
    p = p_new #If not, it fixes the new point as p
  return "Error: Iterations exceeded. The method did not converge"


"""Example Cases"""

if __name__ == "__main__":

    tol = 10e-5
    max_iter = 100000000
    p0 = 0

    f = lambda x: np.sqrt(x+1)  # Example function 1

    root = fixed_point(f,p0,tol,max_iter)
    print(root)  

# ==========================
  # Extra: Plotting section
    # ==========================
    label = "f(x) = sqrt(x+1)" # Example label for the plot

    # --- Plotting ---
    x_vals = np.linspace(0, 4, 400)
    y_vals = f(x_vals)

    plt.figure(figsize=(6, 5))
    plt.plot(x_vals, y_vals, label=label)   # Plot the function
    plt.plot(x_vals, x_vals, 'k--', label="y = x")  # Identity line y=x

    # If a fixed point was found, mark it in red
    if root is not None:
        plt.scatter(root, f(root), color="red", zorder=5, label=f"Punto fijo ≈ {root:.4f}")

    # Plot configuration
    plt.title(f"Método de punto fijo: {label}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()
