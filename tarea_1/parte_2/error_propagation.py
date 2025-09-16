#!/usr/bin/env python3
"""
Error propagation in arithmetic operations.

This module defines a collection of functions for analysing how measurement
errors propagate through basic arithmetic operations such as addition,
subtraction, multiplication and division.  The functions take exact input
values along with their associated absolute errors and compute intervals for
the possible results as well as estimates of the absolute and relative error.

When a measurement a is quoted with an absolute error delta_a, 
the true value lies somewhere in the interval
[a - delta_a, a + delta_a].  For sums and differences the worst-case absolute error in
the result is the sum of the input absolute errors.  For products and
quotients the relative error of the result is bounded by a combination of
relative errors of the inputs; for small uncertainties the product of
uncertainties can usually be neglected.

At the bottom of the file a demonstration script constructs tables that
compare exact results with approximate results under different sign choices
for the uncertainties.  

"""

# Standard library import for basic mathematical functions and constants.
import math
# Pandas is used in the demonstration to organise results into tables. 
import pandas as pd
# Matplotlib can be used to plot results visually. 
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Basic utilities
#
# The functions in this section provide simple helpers for constructing
# intervals and deriving error bounds from those intervals.
# -----------------------------------------------------------------------------

def interval(a: float, da: float, clip_zero: bool = False):
    """Return an interval representing all possible values of a quantity.

    Given a exact value a and its uncertainty da, the true value
    lies somewhere in the closed interval [a-da, a+da].  If negative
    values are not physically meaningful (for example, distances), 
    set clip_zero to True to clip the lower bound at zero.

    Parameters
    ----------
    a : float
        exact (central) value of the measured quantity.
    da : float
        Uncertainty associated with a.  Must be non-negative.
    clip_zero : bool, optional
        If True, negative lower bounds are replaced with 0.0.

    Returns
    -------
    tuple
        A pair (low, high) representing the minimum and maximum possible
        values.  The order is always low ≤ high.

    Raises
    ------
    ValueError
        If da is negative.
    """
    if da < 0:
        raise ValueError("Uncertainty da must be non-negative.")
    low = a - da
    high = a + da
    if clip_zero:
        # For quantities that cannot be negative, discard the negative part.
        low = max(0.0, low)
    return low, high


# -----------------------------------------------------------------------------
# Interval arithmetic for elementary operations
#
# The following functions derive the range of possible results when adding,
# subtracting, multiplying or dividing two measured quantities. Each function
# returns the exact central value, the endpoints of the interval.
# -----------------------------------------------------------------------------

def interval_sum(a: float, b: float, da: float, db: float):
    """Interval and error bounds for ``z = a + b``.

    The result of adding two quantities ``a`` and ``b`` with absolute errors
    ``±da`` and ``±db`` lies in the interval obtained by adding the minimum
    possible values and the maximum possible values, respectively.  The
    absolute error bound is simply ``da + db``, consistent with the theory
    introduced in class.

    Returns
    -------
    tuple
        ``(zc, zmin, zmax, abs_bound, rel_bound)`` where ``zc`` is the exact
        sum ``a + b``, ``[zmin, zmax]`` is the interval of all possible sums,
        and ``abs_bound`` and ``rel_bound`` are the absolute and relative
        error bounds.
    """
    amin, amax = interval(a, da)
    bmin, bmax = interval(b, db)
    zmin = amin + bmin
    zmax = amax + bmax
    zc = a + b
    return zc, zmin, zmax


def interval_difference(a: float, b: float, da: float, db: float):
    """Interval and error bounds for ``z = a − b``.

    The difference ``a − b`` is equivalent to adding ``a`` and ``−b``.  To
    minimise and maximise the result, the smallest possible ``a`` is combined
    with the largest possible ``b`` (for the lower bound) and vice versa for
    the upper bound.  The absolute error bound is again ``da + db``.

    Returns
    -------
    tuple
        ``(zc, zmin, zmax, abs_bound, rel_bound)`` analogous to
        :func:`interval_sum`.
    """
    amin, amax = interval(a, da)
    bmin, bmax = interval(b, db)
    # The smallest possible difference occurs when 'a' is at its minimum and
    # 'b' is at its maximum.  The largest possible difference occurs when
    # 'a' is at its maximum and 'b' is at its minimum.
    zmin = amin - bmax
    zmax = amax - bmin
    zc = a - b
    return zc, zmin, zmax


def interval_product(a: float, b: float, da: float, db: float):
    """Interval and error bounds for ``z = a × b``.

    The product of two quantities with uncertainties requires considering all
    combinations of the extreme values of ``a`` and ``b``.  For four
    combinations of ``a ± da`` and ``b ± db``, the minimum and maximum
    products give the interval.  The absolute and relative error bounds are
    derived from this interval.

    Returns
    -------
    tuple
        ``(zc, zmin, zmax, abs_bound, rel_bound)`` analogous to
        :func:`interval_sum`.
    """
    amin, amax = interval(a, da)
    bmin, bmax = interval(b, db)
    # The distributive property shows that the extreme products occur at the
    # corners of the rectangle defined by the interval endpoints.
    candidates = [amin * bmin, amin * bmax, amax * bmin, amax * bmax]
    zmin = min(candidates)
    zmax = max(candidates)
    zc = a * b
    return zc, zmin, zmax


def interval_quotient(a: float, b: float, da: float, db: float):
    """Interval and error bounds for ``z = a ÷ b``.

    Division by a quantity that may include zero in its interval is undefined
    because the result can become arbitrarily large.  If the interval for
    ``b ± db`` crosses zero, the function returns infinite bounds to indicate
    that no finite interval contains all possible quotients.  Otherwise the
    interval endpoints are found by considering all combinations of ``a ± da``
    and ``b ± db``.

    Returns
    -------
    tuple
        ``(zc, zmin, zmax, abs_bound, rel_bound)`` analogous to
        :func:`interval_sum`.  If division by zero is possible, ``zmin`` and
        ``zmax`` are ``-∞`` and ``+∞``, respectively, and both error bounds
        are infinite.
    """
    amin, amax = interval(a, da)
    bmin, bmax = interval(b, db)
    # If zero lies inside the interval for b, the quotient can be arbitrarily
    # large in magnitude and there is no finite bound.
    if bmin <= 0.0 <= bmax:
        zc = (a / b) if b != 0 else math.nan
        return zc, -math.inf, math.inf, math.inf, math.inf
    # Otherwise, evaluate all four combinations of extremes.
    candidates = [amin / bmin, amin / bmax, amax / bmin, amax / bmax]
    zmin = min(candidates)
    zmax = max(candidates)
    zc = a / b
    return zc, zmin, zmax


# -----------------------------------------------------------------------------
# Error propagation formulas
#
# These functions compute an exact result and an approximate result based on
# exact values and their uncertainties. They return both the absolute and relative errors 
# as well as theoretical bounds derived.  
# -----------------------------------------------------------------------------

def error_propagation_sum(a: float, b: float, delta_a: float, delta_b: float):
    """Propagate errors through a sum a + b.

    The maximum possible deviation of the sum is the sum of the absolute
    uncertainties of the operands.  This function computes the exact sum,
    constructs a worst‑case approximate sum using ``(a + δa) + (b + δb)``, and
    returns both the observed and theoretical errors.

    Returns
    -------
    tuple
        ``(exact_value, approx_value, abs_bound, abs_error, rel_error)`` where
        the last three elements are the theoretical bound, the observed
        absolute error and the observed relative error, respectively.  If the
        exact sum is zero, the relative error is reported as ``math.inf``.
    """
    exact_value = a + b
    approx_value = (a + delta_a) + (b + delta_b)
    abs_bound = delta_a + delta_b
    abs_error = abs(approx_value - exact_value)
    if exact_value != 0:
        rel_error = abs_error / abs(exact_value)
    else:
        rel_error = math.inf
    return exact_value, approx_value, abs_bound, abs_error, rel_error


def error_propagation_difference(a: float, b: float, delta_a: float, delta_b: float) -> tuple:
    """Propagate errors through a difference ``a − b``.

    Although subtraction can reduce the result, the worst‑case absolute error
    bound is still the sum of the individual absolute uncertainties.  This
    function follows the same pattern as :func:`error_propagation_sum` but
    constructs the approximate result using ``(a + δa) − (b + δb)``.

    Returns
    -------
    tuple
        ``(exact_value, approx_value, abs_bound, abs_error, rel_error)`` as in
        :func:`error_propagation_sum`.
    """
    exact_value = a - b
    approx_value = (a + delta_a) - (b + delta_b)
    abs_bound = delta_a + delta_b
    abs_error = abs(approx_value - exact_value)
    if exact_value != 0:
        rel_error = abs_error / abs(exact_value)
    else:
        rel_error = math.inf
    return exact_value, approx_value, abs_bound, abs_error, rel_error


def error_propagation_product(a: float, b: float, delta_a: float, delta_b: float) -> tuple:
    """Propagate errors through a product ``a × b``.

    When multiplying two quantities, the first‑order approximation for the
    absolute error of the product is ``|b|·δa + |a|·δb``.  The exact product
    and an approximate product using ``(a + δa) × (b + δb)`` are computed to
    determine the observed error.  The term ``δa·δb`` is typically very small
    compared with the other terms and is ignored in the bound.

    Returns
    -------
    tuple
        ``(exact_value, approx_value, abs_bound, abs_error, rel_error)`` as in
        :func:`error_propagation_sum`.
    """
    exact_value = a * b
    approx_value = (a + delta_a) * (b + delta_b)
    abs_bound = abs(b) * delta_a + abs(a) * delta_b
    abs_error = abs(approx_value - exact_value)
    if exact_value != 0:
        rel_error = abs_error / abs(exact_value)
    else:
        rel_error = math.inf
    return exact_value, approx_value, abs_bound, abs_error, rel_error


def error_propagation_quotient(a: float, b: float, delta_a: float, delta_b: float) -> tuple:
    """Propagate errors through a quotient ``a ÷ b``.

    Division requires that the denominator ``b`` be non‑zero.  The approximate
    result is computed from ``(a + δa) / (b + δb)`` provided the perturbed
    denominator is non‑zero.  The first‑order approximation for the absolute
    error bound of the quotient is ``(1/|b|)·δa + |a|/|b|²·δb``.  If the exact
    quotient is zero, the relative error is reported as ``math.inf``.

    Returns
    -------
    tuple
        ``(exact_value, approx_value, abs_bound, abs_error, rel_error)`` as in
        :func:`error_propagation_sum`.

    Raises
    ------
    ValueError
        If ``b`` is zero.
    """
    if b == 0:
        raise ValueError("The denominator b must not be zero for division.")
    exact_value = a / b
    if (b + delta_b) != 0:
        approx_value = (a + delta_a) / (b + delta_b)
    else:
        approx_value = math.inf
    abs_bound = (1 / abs(b)) * delta_a + (abs(a) / (abs(b) ** 2)) * delta_b
    abs_error = abs(approx_value - exact_value)
    if exact_value != 0:
        rel_error = abs_error / abs(exact_value)
    else:
        rel_error = math.inf
    return exact_value, approx_value, abs_bound, abs_error, rel_error


# -----------------------------------------------------------------------------
# Demonstration
#
# When this module is run as a script, it prints tables comparing exact and
# approximate results for a couple of examples.  Users can change the sign
# choices to explore how different combinations of positive and negative
# uncertainties affect the outcome.
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Two examples to contrast the effect of the relative error.  The second
    # example uses values that are ten times smaller but the same absolute
    # uncertainties.
    examples = [
        (5.0, 3.0, 0.2, 0.1),
        (0.5, 0.3, 0.2, 0.1),
    ]

    # Formatting utilities for the console output.
    def _fmt_rel_pct(x):
        if x is None or (isinstance(x, float) and not math.isfinite(x)):
            return "N/A"
        return f"{x:.2f}"

    def _fmt_float(x, n: int = 6):
        if isinstance(x, str):
            return x
        if x is None:
            return "N/A"
        if isinstance(x, float) and not math.isfinite(x):
            return "∞"
        return f"{x:.{n}f}"

    def _ascii_table(df: pd.DataFrame, formats: dict | None = None) -> None:
        """Print a pandas DataFrame using ASCII characters for borders."""
        columns = list(df.columns)
        rows = df.to_dict("records")

        def apply_fmt(column, value):
            if formats and column in formats:
                return formats[column](value)
            return value if isinstance(value, str) else str(value)

        # Format each cell according to the provided functions.
        cells = [[apply_fmt(c, r[c]) for c in columns] for r in rows]
        widths = [max(len(str(h)), *(len(str(row[i])) for row in cells)) for i, h in enumerate(columns)]

        def line(char="─", cross="┼", left="├", right="┤"):
            pieces = [char * (w + 2) for w in widths]
            return left + cross.join(pieces) + right

        # Build the ASCII table.
        top = "┌" + "┬".join("─" * (w + 2) for w in widths) + "┐"
        header = "│ " + " │ ".join(str(h).ljust(widths[i]) for i, h in enumerate(columns)) + " │"
        mid = line()
        bottom = "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"

        print(top)
        print(header)
        print(mid)
        for row in cells:
            print("│ " + " │ ".join(str(row[i]).ljust(widths[i]) for i in range(len(columns))) + " │")
        print(bottom)

    def _interval_str(lo: float, hi: float) -> str:
        # Represent infinite intervals explicitly.
        if not (math.isfinite(lo) and math.isfinite(hi)):
            return "(-∞, +∞)"
        return f"[{lo:.6f}, {hi:.6f}]"

    # Iterate over each example and build a table of results.
    for idx, (a, b, da, db) in enumerate(examples, start=1):
        # Compute interval bounds using interval arithmetic.
        z_sum, sum_low, sum_high = interval_sum(a, b, da, db)
        z_diff, diff_low, diff_high = interval_difference(a, b, da, db)
        z_prod, prod_low, prod_high = interval_product(a, b, da, db)
        z_quot, quot_low, quot_high = interval_quotient(a, b, da, db)

        # Compute error propagation results using the functions you created
        exact_sum, approx_sum, sum_bound, sum_abs_error, sum_rel_error = error_propagation_sum(a, b, da, db)
        exact_diff, approx_diff, diff_bound, diff_abs_error, diff_rel_error = error_propagation_difference(a, b, da, db)
        exact_prod, approx_prod, prod_bound, prod_abs_error, prod_rel_error = error_propagation_product(a, b, da, db)
        exact_quot, approx_quot, quot_bound, quot_abs_error, quot_rel_error = error_propagation_quotient(a, b, da, db)

        # Construct a list of dictionaries for each operation.
        rows = [
            {
                "Operation": "Sum",
                "a ± δa": f"{a:g} ± {da:g}",
                "b ± δb": f"{b:g} ± {db:g}",
                "Exact value": exact_sum,
                "Approximate value": approx_sum,
                "Absolute error": sum_abs_error,
                "Relative error (%)": sum_rel_error * 100 if math.isfinite(sum_rel_error) else None,
                "Theoretical bound": sum_bound,
                "Possible results [min, max]": _interval_str(sum_low, sum_high)
            },
            {
                "Operation": "Difference",
                "a ± δa": f"{a:g} ± {da:g}",
                "b ± δb": f"{b:g} ± {db:g}",
                "Exact value": exact_diff,
                "Approximate value": approx_diff,
                "Absolute error": diff_abs_error,
                "Relative error (%)": diff_rel_error * 100 if math.isfinite(diff_rel_error) else None,
                "Theoretical bound": diff_bound,
                "Possible results [min, max]": _interval_str(diff_low, diff_high)
            },
            {
                "Operation": "Product",
                "a ± δa": f"{a:g} ± {da:g}",
                "b ± δb": f"{b:g} ± {db:g}",
                "Exact value": exact_prod,
                "Approximate value": approx_prod,
                "Absolute error": prod_abs_error,
                "Relative error (%)": prod_rel_error * 100 if math.isfinite(prod_rel_error) else None,
                "Theoretical bound": prod_bound,
                "Possible results [min, max]": _interval_str(prod_low, prod_high)
            },
            {
                "Operation": "Quotient",
                "a ± δa": f"{a:g} ± {da:g}",
                "b ± δb": f"{b:g} ± {db:g}",
                "Exact value": exact_quot,
                "Approximate value": approx_quot,
                "Absolute error": quot_abs_error,
                "Relative error (%)": quot_rel_error * 100 if math.isfinite(quot_rel_error) else None,
                "Theoretical bound": quot_bound,
                "Possible results [min, max]": _interval_str(quot_low, quot_high)
            },
        ]

        df = pd.DataFrame(rows)
        
        # Define the order of columns for readability.
        column_order = [
            "Operation",
            "a ± δa",
            "b ± δb",
            "Exact value",
            "Approximate value",
            "Absolute error",
            "Relative error (%)",
            "Theoretical bound",
            "Possible results [min, max]"
        ]
        df = df[column_order]

        # Define formatting functions for each column.
        formats = {
            "Exact value": lambda x: _fmt_float(x, 6),
            "Approximate value": lambda x: _fmt_float(x, 6),
            "Absolute error": lambda x: _fmt_float(x, 6),
            "Relative error (%)": _fmt_rel_pct,
            "Theoretical bound": lambda x: _fmt_float(x, 6)
        }

        print(f"\n=== TABLE — Example {idx} ===")
        print("• a ± δa, b ± δb: measured value and its absolute error (uncertainty).")
        print("• Exact value: the result using exact values a and b.")
        print("• Approximate value: the result using (a + δa) and (b + δb).")
        print("• Absolute error: |Approximate value − Exact value|.")
        print("• Relative error (%): Absolute error divided by |Exact value| × 100%.")
        print("• Theoretical bound: maximum possible absolute error from propagation theory.")
        print("• Possible results [min, max]: complete range considering all possible values.")
        print()

        _ascii_table(df, formats=formats)