#!/usr/bin/env python3
"""
Error propagation in arithmetic operations.

This module defines a collection of functions for analysing how measurement
errors propagate through basic arithmetic operations such as addition,
subtraction, multiplication and division.  The functions take nominal input
values along with their associated absolute errors and compute intervals for
the possible results as well as estimates of the absolute and relative error.

The underlying theory is simple: when a measurement `a` is quoted with an
absolute error `δa`, the true value lies somewhere in the interval
`[a − δa, a + δa]`.  For sums and differences the worst‑case absolute error in
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

def interval(a: float, da: float, clip_zero: bool = False) -> tuple:
    """Return an interval representing all possible values of a quantity.

    Given a nominal value ``a`` and its absolute error ``da``, the true value
    lies somewhere in the closed interval ``[a − da, a + da]``.  If negative
    values are not physically meaningful (for example, distances or
    concentrations), set ``clip_zero`` to ``True`` to clip the lower bound at
    zero.

    Parameters
    ----------
    a : float
        Nominal (central) value of the measured quantity.
    da : float
        Absolute error associated with ``a``.  Must be non‑negative.
    clip_zero : bool, optional
        If ``True``, negative lower bounds are replaced with ``0.0``.

    Returns
    -------
    tuple
        A pair ``(low, high)`` representing the minimum and maximum possible
        values.  The order is always ``low ≤ high``.

    Raises
    ------
    ValueError
        If ``da`` is negative.
    """
    if da < 0:
        raise ValueError("Absolute error da must be non‑negative.")
    low = a - da
    high = a + da
    if clip_zero:
        # For quantities that cannot be negative, discard the negative part.
        low = max(0.0, low)
    return low, high


def bounds_from_interval(zc: float, zmin: float, zmax: float) -> tuple:
    """Compute absolute and relative error bounds from a result interval.

    When the exact value of a result is ``zc`` and the possible result lies in
    the interval ``[zmin, zmax]``, the worst‑case absolute error is the maximum
    of the deviations from the central value.  The relative error is defined
    as the absolute error divided by the magnitude of the exact value, and
    yields ``None`` if the exact value is zero (to avoid division by zero).

    Parameters
    ----------
    zc : float
        Exact (central) value of the result.
    zmin : float
        Lower end of the interval of possible results.
    zmax : float
        Upper end of the interval of possible results.

    Returns
    -------
    tuple
        A tuple ``(abs_bound, rel_bound)`` where ``abs_bound`` is the maximum
        absolute error and ``rel_bound`` is the relative error (or ``None`` if
        the exact value is zero).
    """
    abs_bound = max(abs(zc - zmin), abs(zmax - zc))
    if zc == 0:
        rel_bound = None  # Relative error is undefined when the exact value is zero
    else:
        rel_bound = abs_bound / abs(zc)
    return abs_bound, rel_bound


# -----------------------------------------------------------------------------
# Interval arithmetic for elementary operations
#
# The following functions derive the range of possible results when adding,
# subtracting, multiplying or dividing two measured quantities.  Each function
# returns the exact central value, the endpoints of the interval, and the
# corresponding absolute and relative error bounds.
# -----------------------------------------------------------------------------

def interval_sum(a: float, b: float, da: float, db: float) -> tuple:
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
    abs_bound, rel_bound = bounds_from_interval(zc, zmin, zmax)
    return zc, zmin, zmax, abs_bound, rel_bound


def interval_difference(a: float, b: float, da: float, db: float) -> tuple:
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
    abs_bound, rel_bound = bounds_from_interval(zc, zmin, zmax)
    return zc, zmin, zmax, abs_bound, rel_bound


def interval_product(a: float, b: float, da: float, db: float) -> tuple:
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
    abs_bound, rel_bound = bounds_from_interval(zc, zmin, zmax)
    return zc, zmin, zmax, abs_bound, rel_bound


def interval_quotient(a: float, b: float, da: float, db: float) -> tuple:
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
    abs_bound, rel_bound = bounds_from_interval(zc, zmin, zmax)
    return zc, zmin, zmax, abs_bound, rel_bound


# -----------------------------------------------------------------------------
# Error propagation formulas
#
# These functions compute an exact result and an approximate result based on
# nominal values and their uncertainties.  They return both the observed
# absolute and relative errors as well as theoretical bounds derived from
# first‑order approximations.  These functions illustrate how to propagate
# uncertainties through basic operations without resorting to interval
# arithmetic.
# -----------------------------------------------------------------------------

def error_propagation_sum(a: float, b: float, delta_a: float, delta_b: float) -> tuple:
    """Propagate errors through a sum ``a + b``.

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
    # Sign choices for the approximate values.  The two characters in
    # SELECTED_CASE correspond to whether the error is added (+) or subtracted
    # (−) from each of the two operands.  For example ("+", "−") means a + δa
    # and b − δb.
    SELECTED_CASE = ("+", "+")  # Change this tuple to explore other cases
    CASE_LABEL = f"{SELECTED_CASE[0]},{SELECTED_CASE[1]}"
    VAL_AP_COL = f"Approximate value (case {CASE_LABEL})"
    ERR_ABS_COL = f"Absolute error (case {CASE_LABEL})"
    ERR_REL_COL = f"Relative error (case {CASE_LABEL}) (%)"

    # Two examples to contrast the effect of the relative error.  The second
    # example uses values that are ten times smaller but the same absolute
    # uncertainties.
    examples = [
        (5.0, 3.0, 0.2, 0.1),
        (0.5, 0.3, 0.2, 0.1),
    ]

    # Formatting utilities for the console output.  These small helper
    # functions format numbers consistently and handle special cases like
    # non‑finite values or None.
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
        """Print a pandas DataFrame using ASCII characters for borders.

        The optional ``formats`` dictionary maps column names to formatting
        functions that take a cell value and return a string.  This avoids
        repeating format code when printing the same DataFrame in different
        contexts.
        """
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

    # Helpers for constructing approximate values based on the selected signs.
    def _apply_sign(x: float, dx: float, sign: str) -> float:
        return x + dx if sign == "+" else x - dx

    def _evaluate_case_operation(operation: str, a: float, b: float, da: float, db: float, sign_a: str, sign_b: str) -> tuple:
        """Return the approximate value and errors for a chosen sign case.

        This internal helper performs the specified operation using ``a ± δa`` and
        ``b ± δb`` according to the supplied signs.  It then computes the exact
        result and the absolute and relative errors.  If the approximate result
        is infinite or not finite, the errors are set accordingly.
        """
        A = _apply_sign(a, da, sign_a)
        B = _apply_sign(b, db, sign_b)
        if operation == "Sum":
            exact, approx = a + b, A + B
        elif operation == "Difference":
            exact, approx = a - b, A - B
        elif operation == "Product":
            exact, approx = a * b, A * B
        elif operation == "Quotient":
            exact = a / b
            approx = A / B if B != 0 else math.inf
        else:
            raise ValueError("Unknown operation.")
        abs_error = abs(approx - exact) if math.isfinite(approx) else math.inf
        rel_error = (abs_error / abs(exact)) if exact != 0 and math.isfinite(abs_error) else math.inf
        return approx, abs_error, rel_error

    # Iterate over each example and build a table of results for the selected case.
    for idx, (a, b, da, db) in enumerate(examples, start=1):
        # Compute interval bounds using interval arithmetic.
        z_sum, sum_low, sum_high, sum_abs_bound, sum_rel_bound = interval_sum(a, b, da, db)
        z_diff, diff_low, diff_high, diff_abs_bound, diff_rel_bound = interval_difference(a, b, da, db)
        z_prod, prod_low, prod_high, prod_abs_bound, prod_rel_bound = interval_product(a, b, da, db)
        z_quot, quot_low, quot_high, quot_abs_bound, quot_rel_bound = interval_quotient(a, b, da, db)

        # Compute approximate values and errors for the chosen sign combination.
        sum_ap, sum_ea, sum_er = _evaluate_case_operation("Sum", a, b, da, db, *SELECTED_CASE)
        diff_ap, diff_ea, diff_er = _evaluate_case_operation("Difference", a, b, da, db, *SELECTED_CASE)
        prod_ap, prod_ea, prod_er = _evaluate_case_operation("Product", a, b, da, db, *SELECTED_CASE)
        quot_ap, quot_ea, quot_er = _evaluate_case_operation("Quotient", a, b, da, db, *SELECTED_CASE)

        def _interval_str(lo: float, hi: float) -> str:
            # Represent infinite intervals explicitly.
            if not (math.isfinite(lo) and math.isfinite(hi)):
                return "(-∞, +∞)"
            return f"[{lo:.6f}, {hi:.6f}]"

        # Construct a list of dictionaries for each operation.
        rows = [
            {
                "Operation": "Sum",
                "a ± δa": f"{a:g} ± {da:g}",
                "b ± δb": f"{b:g} ± {db:g}",
                "Exact value": z_sum,
                VAL_AP_COL: sum_ap,
                ERR_ABS_COL: sum_ea,
                ERR_REL_COL: sum_er * 100 if math.isfinite(sum_er) else None,
                "Possible results [min, max]": _interval_str(sum_low, sum_high),
                "Absolute error bound": sum_abs_bound,
                "Relative error bound (%)": None if sum_rel_bound is None else (sum_rel_bound * 100),
            },
            {
                "Operation": "Difference",
                "a ± δa": f"{a:g} ± {da:g}",
                "b ± δb": f"{b:g} ± {db:g}",
                "Exact value": z_diff,
                VAL_AP_COL: diff_ap,
                ERR_ABS_COL: diff_ea,
                ERR_REL_COL: diff_er * 100 if math.isfinite(diff_er) else None,
                "Possible results [min, max]": _interval_str(diff_low, diff_high),
                "Absolute error bound": diff_abs_bound,
                "Relative error bound (%)": None if diff_rel_bound is None else (diff_rel_bound * 100),
            },
            {
                "Operation": "Product",
                "a ± δa": f"{a:g} ± {da:g}",
                "b ± δb": f"{b:g} ± {db:g}",
                "Exact value": z_prod,
                VAL_AP_COL: prod_ap,
                ERR_ABS_COL: prod_ea,
                ERR_REL_COL: prod_er * 100 if math.isfinite(prod_er) else None,
                "Possible results [min, max]": _interval_str(prod_low, prod_high),
                "Absolute error bound": prod_abs_bound,
                "Relative error bound (%)": None if prod_rel_bound is None else (prod_rel_bound * 100),
            },
            {
                "Operation": "Quotient",
                "a ± δa": f"{a:g} ± {da:g}",
                "b ± δb": f"{b:g} ± {db:g}",
                "Exact value": z_quot,
                VAL_AP_COL: quot_ap,
                ERR_ABS_COL: quot_ea,
                ERR_REL_COL: quot_er * 100 if math.isfinite(quot_er) else None,
                "Possible results [min, max]": _interval_str(quot_low, quot_high),
                "Absolute error bound": quot_abs_bound,
                "Relative error bound (%)": None if (quot_rel_bound is None or not math.isfinite(quot_rel_bound)) else (quot_rel_bound * 100),
            },
        ]

        df = pd.DataFrame(rows)
        # Define the order of columns for readability.
        column_order = [
            "Operation",
            "a ± δa",
            "b ± δb",
            "Exact value",
            VAL_AP_COL,
            ERR_ABS_COL,
            ERR_REL_COL,
            "Possible results [min, max]",
            "Absolute error bound",
            "Relative error bound (%)",
        ]
        df = df[column_order]

        # Define formatting functions for each column.
        formats = {
            "Exact value": lambda x: _fmt_float(x, 6),
            VAL_AP_COL: lambda x: _fmt_float(x, 6),
            ERR_ABS_COL: lambda x: _fmt_float(x, 6),
            ERR_REL_COL: _fmt_rel_pct,
            "Absolute error bound": lambda x: _fmt_float(x, 6),
            "Relative error bound (%)": _fmt_rel_pct,
        }

        print(f"\n=== TABLE — Example {idx} ===")
        # Explanation of the columns in plain language without jargon.
        print("• a ± δa, b ± δb: measured value and its ± absolute error (uncertainty).")
        print("• Exact value: the result using a and b without applying ±δ.")
        print(f"• {VAL_AP_COL}: the result using a±δa and b±δb according to the chosen signs.")
        print(f"• {ERR_ABS_COL}: |Approximate value − Exact value|.")
        print(f"• {ERR_REL_COL}: Absolute error divided by |Exact value| × 100%.")
        print("• Possible results [min, max]: taking any a between (a−δa and a+δa) ")
        print("  and any b between (b−δb and b+δb), the result of the operation always")
        print("  lies between those two numbers (the smallest and largest possible).")
        print("• Absolute error bound: the maximum deviation of the result due to the ±δ inputs.")
        print("• Relative error bound (%): the absolute error bound divided by |Exact value|, expressed as a percentage.\n")

        _ascii_table(df, formats=formats)