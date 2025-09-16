#!/usr/bin/env python3
"""
Error propagation in arithmetic operations.

This module defines a collection of functions for analysing how measurement
errors propagate through basic arithmetic operations such as addition,
subtraction, multiplication and division. The functions take exact input
values along with their associated uncertainties and compute estimates of 
the absolute and relative error using standard error propagation formulas.

When a measurement a is quoted with an uncertainty delta_a, 
the true value lies somewhere in the interval
[a - delta_a, a + delta_a].  For sums and differences the worst-case absolute error in
the result is the sum of the input absolute errors.  For products and
quotients the relative error of the result is bounded by a combination of
relative errors of the inputs; for small uncertainties the product of
uncertainties can usually be neglected.

"""

# Standard library import for basic mathematical functions and constants.
import math
# Pandas is used in the demonstration to organise results into tables. 
import pandas as pd

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
    constructs a worst-case approximate sum using (a + delta_a) + (b + delta_b), and
    returns both the observed and theoretical errors.

    Returns
    -------
    tuple
        (exact_value, approx_value, abs_bound, abs_error, rel_error) where
        the last three elements are the absolute error bound, the observed
        absolute error and the observed relative error, respectively.  If the
        exact sum is zero, the relative error is reported as math.inf.
    """
    exact_value = a + b
    approx_value = (a + delta_a) + (b + delta_b)
    abs_bound = abs(delta_a) + abs(delta_b)
    abs_error = abs(approx_value - exact_value)
    if exact_value != 0:
        rel_error = abs_error / abs(exact_value)
    else:
        rel_error = math.inf
    return exact_value, approx_value, abs_bound, abs_error, rel_error


def error_propagation_difference(a: float, b: float, delta_a: float, delta_b: float):
    """Propagate errors through a difference a - b.

    Although subtraction can reduce the result, the worst-case absolute error
    bound is still the sum of the individual absolute uncertainties.  

    Returns
    -------
    tuple
        (exact_value, approx_value, abs_bound, abs_error, rel_error) where
        the last three elements are the absolute error bound, the observed
        absolute error and the observed relative error, respectively.  If the
        exact sum is zero, the relative error is reported as math.inf.
    """
    exact_value = a - b
    approx_value = (a + delta_a) - (b + delta_b)
    abs_bound = abs(delta_a) + abs(delta_b)
    abs_error = abs(approx_value - exact_value)
    if exact_value != 0:
        rel_error = abs_error / abs(exact_value)
    else:
        rel_error = math.inf
    return exact_value, approx_value, abs_bound, abs_error, rel_error


def error_propagation_product(a: float, b: float, delta_a: float, delta_b: float):
    """Propagate errors through a product a x b.

    When multiplying two quantities, the first-order approximation for the
    absolute error of the product is |b|·delta_a + |a|·delta_b.  The exact product
    and an approximate product using (a + delta_a) x (b + delta_b) are computed to
    determine the observed error.  The term delta_a·delta_b is typically very small
    compared with the other terms and is ignored in the bound.

    Returns
    -------
    tuple
        (exact_value, approx_value, abs_bound, abs_error, rel_error, rel_bound) where
        the last four elements are the absolute error bound, the observed
        absolute error and the observed relative error, relative error bound respectively.  If the
        exact sum is zero, the relative error is reported as math.inf.
    """
    exact_value = a * b
    approx_value = (a + delta_a) * (b + delta_b)

    abs_bound = abs(b * delta_a) + abs(a * delta_b)
    rel_bound = (abs(delta_a) / abs(a) if a != 0 else 0) + (abs(delta_b) / abs(b) if b != 0 else 0)
    abs_error = abs(approx_value - exact_value)
    if exact_value != 0:
        rel_error = abs_error / abs(exact_value)
    else:
        rel_error = math.inf
    return exact_value, approx_value, abs_bound, abs_error, rel_error, rel_bound


def error_propagation_quotient(a: float, b: float, delta_a: float, delta_b: float):
    """Propagate errors through a quotient a ÷ b.

    Division requires that the denominator b be non-zero.  The approximate
    result is computed from (a + delta_a) / (b + delta_b) provided the perturbed
    denominator is non-zero.  The first-order approximation for the absolute
    error bound of the quotient is (1/|b|)·delta_a + |a|/|b|^2·delta_b.  If the exact
    quotient is zero, the relative error is reported as math.inf.

    Returns
    -------
    tuple
        (exact_value, approx_value, abs_bound, abs_error, rel_error, rel_bound) where
        the last three elements are the absolute error bound, the observed
        absolute error and the observed relative error, respectively.  If the
        exact sum is zero, the relative error is reported as math.inf.

    Raises
    ------
    ValueError
        If b is zero.
    """
    if b == 0:
        raise ValueError("The denominator b must not be zero for division.")
    exact_value = a / b
    if (b + delta_b) != 0:
        approx_value = (a + delta_a) / (b + delta_b)
    else:
        approx_value = math.inf
    abs_bound = abs(delta_a/b) + abs(a * delta_b / (b**2))
    rel_bound = (abs(delta_a) / abs(a) if a != 0 else 0) + (abs(delta_b) / abs(b) if b != 0 else 0)

    abs_error = abs(approx_value - exact_value)
    if exact_value != 0:
        rel_error = abs_error / abs(exact_value)
    else:
        rel_error = math.inf
    return exact_value, approx_value, abs_bound, abs_error, rel_error, rel_bound


# -----------------------------------------------------------------------------
# Demonstration
#
# When this module is run as a script, it prints tables comparing exact and
# approximate results for a couple of examples.
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Example values with uncertainties
    examples = [
        (5.0, 3.0, 0.2, 0.1),
        (0.5, 0.3, 0.2, 0.1),
    ]

    # Iterate over each example and build a table of results.
    for idx, (a, b, da, db) in enumerate(examples, start=1):
        # Compute error propagation results using the functions defined above
        exact_sum, approx_sum, sum_bound, sum_abs_error, sum_rel_error = error_propagation_sum(a, b, da, db)
        exact_diff, approx_diff, diff_bound, diff_abs_error, diff_rel_error = error_propagation_difference(a, b, da, db)
        exact_prod, approx_prod, prod_bound, prod_abs_error, prod_rel_error, prod_rel_bound = error_propagation_product(a, b, da, db)
        exact_quot, approx_quot, quot_bound, quot_abs_error, quot_rel_error, quot_rel_bound = error_propagation_quotient(a, b, da, db)

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
                "Absolute error bound": sum_bound,
                "Relative error bound": None,
            },
            {
                "Operation": "Difference",
                "a ± δa": f"{a:g} ± {da:g}",
                "b ± δb": f"{b:g} ± {db:g}",
                "Exact value": exact_diff,
                "Approximate value": approx_diff,
                "Absolute error": diff_abs_error,
                "Relative error (%)": diff_rel_error * 100 if math.isfinite(diff_rel_error) else None,
                "Absolute error bound": diff_bound,
                "Relative error bound": None,
            },
            {
                "Operation": "Product",
                "a ± δa": f"{a:g} ± {da:g}",
                "b ± δb": f"{b:g} ± {db:g}",
                "Exact value": exact_prod,
                "Approximate value": approx_prod,
                "Absolute error": prod_abs_error,
                "Relative error (%)": prod_rel_error * 100 if math.isfinite(prod_rel_error) else None,
                "Absolute error bound": prod_bound,
                "Relative error bound": prod_rel_bound,
            },
            {
                "Operation": "Quotient",
                "a ± δa": f"{a:g} ± {da:g}",
                "b ± δb": f"{b:g} ± {db:g}",
                "Exact value": exact_quot,
                "Approximate value": approx_quot,
                "Absolute error": quot_abs_error,
                "Relative error (%)": quot_rel_error * 100 if math.isfinite(quot_rel_error) else None,
                "Absolute error bound": quot_bound,
                "Relative error bound": quot_rel_bound,
            },
        ]
        # Create DataFrame 
        df = pd.DataFrame(rows)
        column_order = [
            "Operation",
            "a ± δa",
            "b ± δb",
            "Exact value",
            "Approximate value",
            "Absolute error",
            "Relative error (%)",
            "Absolute error bound",
            "Relative error bound",
        ]
        df = df[column_order]

        # Print explanatory notes
        print(f"\n=== TABLE — Example {idx} ===")
        print("• a ± δa, b ± δb: measured value and its absolute error (uncertainty).")
        print("• Exact value: the result using exact values a and b.")
        print("• Approximate value: the result using (a + δa) and (b + δb).")
        print("• Absolute error: |Approximate value − Exact value|.")
        print("• Relative error (%): Absolute error divided by |Exact value| × 100%.")
        print("• Absolute error bound: maximum possible absolute error from propagation error.")
        print("• Relative error bound: maximum possible relative error from propagation error.")
        print()

        float_cols = [
            "Exact value",
            "Approximate value",
            "Absolute error",
            "Relative error (%)",
            "Absolute error bound",
            "Relative error bound",
        ]
        # Make a copy to avoid modifying the original DataFrame
        df_display = df.copy()
        for col in float_cols:
            # Only attempt to round numeric columns; non-numeric entries (None) are left as-is
            df_display[col] = df_display[col].apply(
                lambda x: round(x, 6) if isinstance(x, (int, float)) and math.isfinite(x) else x
            )
        # Reset index so that the DataFrame prints without the default numeric index
        df_display.reset_index(drop=True, inplace=True)
        print(df_display.to_string(index=False))