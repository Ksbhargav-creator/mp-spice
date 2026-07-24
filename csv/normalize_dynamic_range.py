"""Normalize csv/dynamic_range_histogram.csv into per-bucket percentages and
per-run summary statistics, so histograms with different iteration counts
(and therefore different total dot-product samples) can be compared directly
instead of eyeballing raw bar heights across runs.

Run from inside the csv/ directory:

    python3 normalize_dynamic_range.py

Reads:  dynamic_range_histogram.csv
Writes: dynamic_range_normalized.csv  -- same rows as the input, plus a
                                          Percentage column (Count / Total
                                          for that run, as a %).
        dynamic_range_summary.csv    -- one row per (Matrix, Experiment,
                                          Type, Variant) run: weighted
                                          Mean/Median/P90/Max dynamic range
                                          in decades, plus Total/
                                          SingleTermRows/EmptyRows for
                                          context.
"""
import numpy as np
import pandas as pd

GROUP_COLS = ["Matrix", "n", "Experiment", "Type", "Variant"]


def weighted_percentile(decades, counts, pct):
    """Weighted percentile (0-100) over bucketed (decade, count) data for
    ONE run. Buckets are already integer decade bins, so this finds the
    bucket where the cumulative count first reaches `pct`% of the total --
    the natural definition of a percentile over binned/grouped data."""
    order = np.argsort(decades)
    d = np.asarray(decades)[order]
    c = np.asarray(counts)[order]
    total = c.sum()
    if total == 0:
        return np.nan
    cum = np.cumsum(c)
    target = pct / 100.0 * total
    idx = int(np.searchsorted(cum, target))
    idx = min(idx, len(d) - 1)
    return float(d[idx])


def weighted_mean(decades, counts):
    total = counts.sum()
    if total == 0:
        return np.nan
    return float((decades * counts).sum() / total)


def main():
    df = pd.read_csv("dynamic_range_histogram.csv")

    # The C++ writer opens this file in append mode, so re-running the study
    # binary against the same matrix adds a second, fully-identical copy of
    # its rows rather than replacing them. Drop those before aggregating --
    # otherwise Count gets silently double-counted per rerun while Total
    # (recorded per-row, from a single run) doesn't, corrupting every
    # percentage and summary statistic below.
    before = len(df)
    df = df.drop_duplicates()
    if len(df) != before:
        print(f"Dropped {before - len(df)} duplicate rows (likely reruns of the same matrix)")

    # --- normalized (percentage) view: same shape as the input CSV, one
    # extra column. This is what you want for overlaying/comparing bar
    # charts across runs with different iteration counts. ---
    df["Percentage"] = df.groupby(GROUP_COLS)["Count"].transform(
        lambda c: 100.0 * c / c.sum()
    )
    df.to_csv("dynamic_range_normalized.csv", index=False)
    print(f"Wrote dynamic_range_normalized.csv ({len(df)} rows)")

    # --- summary view: one row per run, the numbers actually worth
    # comparing at a glance across many matrices. ---
    rows = []
    for keys, g in df.groupby(GROUP_COLS):
        matrix, n, experiment, type_, variant = keys
        decades = g["RangeDecades"].to_numpy()
        counts = g["Count"].to_numpy()
        rows.append(
            {
                "Matrix": matrix,
                "n": n,
                "Experiment": experiment,
                "Type": type_,
                "Variant": variant,
                "Mean": round(weighted_mean(decades, counts), 3),
                "Median": weighted_percentile(decades, counts, 50),
                "P90": weighted_percentile(decades, counts, 90),
                "Max": int(decades.max()),
                "Total": int(g["Total"].iloc[0]),
                "SingleTermRows": int(g["SingleTermRows"].iloc[0]),
                "EmptyRows": int(g["EmptyRows"].iloc[0]),
            }
        )

    summary = pd.DataFrame(rows).sort_values(GROUP_COLS)
    summary.to_csv("dynamic_range_summary.csv", index=False)
    print(f"Wrote dynamic_range_summary.csv ({len(summary)} rows)")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
