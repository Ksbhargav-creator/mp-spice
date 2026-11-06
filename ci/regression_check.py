#!/usr/bin/env python3
"""mp-spice -- Table 2 (working-precision residual) regression check.

Two independent kinds of checks, deliberately not conflated into one
pass/fail number (see the design discussion this script came out of):

1. STRUCTURAL INVARIANT (hard failure, catches a bug present since day one).
   `posit<32,2>` cannot physically represent a residual/forward-error tighter
   than roughly its own precision floor near magnitude 1. For posit<n,es>,
   the fraction field near 1.0 is approximately n - 1 (sign) - 2 (minimum
   regime) - es bits, so posit<32,2> has ~27 fraction bits -> a relative
   precision floor of ~2^-27 (~7.45e-9). PRECISION_FLOOR below is set an
   order of magnitude looser than that theoretical limit, so only a result
   that is IMPOSSIBLE for genuine posit<32,2> arithmetic trips it -- e.g. a
   result sitting near double's ~1e-16 floor, which is exactly the signature
   the original double-casting bug produced (residual/correction vectors
   silently living at `double` between refinement iterations instead of at
   `Working` precision). This check does NOT apply to posit<64,3>, which is
   close enough to double that tight results are legitimate.

   This check would NOT have been retroactively verifiable against the
   original bug's raw small-matrix logs -- those weren't preserved as
   individual files, only rajat30's pre/post-fix logs were, and rajat30
   itself was never in the "no gain" regime that motivated the bug hunt.
   The check is derived from the bug's mechanism (a type-precision mismatch),
   not curve-fit to a specific historical number.

2. DRIFT VS. BASELINE (soft warning, catches a regression between two known-
   good states). Baseline is a human-approved, git-committed JSON file --
   never auto-updated by a passing run -- compared with a wide tolerance
   band, since real matrix-specific behavior varies legitimately (rajat17 is
   a confirmed case where quire is *supposed* to be worse than plain; a tight
   band would false-alarm on it forever).

Usage:
    python3 regression_check.py --logs logs/rajat11_sweep.log logs/rajat14_sweep.log \
        --baseline ci/baseline.json --history ci/regression_history.csv

Exit code is nonzero only on a structural-invariant violation. Drift
warnings are printed and recorded but do not fail the job -- see the design
note above on why "quire always wins" is the wrong thing to hard-fail on.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

PRECISION_FLOOR = {
    "posit<32,2>": 1e-9,   # ~10x looser than the ~7.45e-9 theoretical fraction-bit limit
    # posit<64,3> intentionally has no floor -- it's close enough to double
    # that tight results are legitimate, not a bug signature.
}

# Matrices known to fail to load (unsupported Matrix Market 'pattern' field).
# These are expected failures, not regressions -- see rajat01/07/08/09/10.
EXPECTED_LOAD_FAILURES = {"rajat01", "rajat07", "rajat08", "rajat09", "rajat10"}

DRIFT_TOLERANCE = 5.0  # current value must stay within [baseline/TOL, baseline*TOL]

ROW_RE = re.compile(
    r"^(posit<\d+,\d+>)\s*\|\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+(\d+)\s*\|\s*"
    r"([\d.eE+-]+)\s+([\d.eE+-]+)\s+(\d+)"
)


def parse_working_precision_table(log_text):
    """Extract the 'Working-precision residual' table rows from one
    klu_quire_IR_study log. Returns {type: {plain_res, plain_ferr, plain_it,
    quire_res, quire_ferr, quire_it}}."""
    lines = log_text.splitlines()
    rows = {}
    in_table = False
    for line in lines:
        if "Working-precision residual" in line:
            in_table = True
            continue
        if not in_table:
            continue
        m = ROW_RE.match(line.strip())
        if m:
            typ, pres, pferr, pit, qres, qferr, qit = m.groups()
            rows[typ] = {
                "plain_res": float(pres), "plain_ferr": float(pferr), "plain_it": int(pit),
                "quire_res": float(qres), "quire_ferr": float(qferr), "quire_it": int(qit),
            }
        elif rows and line.strip() == "":
            break  # end of table
    return rows


def matrix_name_from_log_path(path):
    base = os.path.basename(path)
    return base[: -len("_sweep.log")] if base.endswith("_sweep.log") else base.rsplit(".", 1)[0]


def check_load_failure(log_text, matrix_name):
    crashed = "unsupported Matrix Market field" in log_text or "terminating due to uncaught exception" in log_text
    if crashed:
        if matrix_name in EXPECTED_LOAD_FAILURES:
            return "EXPECTED_FAIL"
        return "UNEXPECTED_FAIL"  # a matrix that used to load no longer does -- a real regression
    return None


def check_structural_invariant(matrix_name, typ, values):
    floor = PRECISION_FLOOR.get(typ)
    if floor is None:
        return []
    violations = []
    for variant in ("plain", "quire"):
        ferr = values[f"{variant}_ferr"]
        if ferr < floor:
            violations.append(
                f"{matrix_name} {typ} {variant}: forward error {ferr:.3e} is below the "
                f"physical precision floor ({floor:.1e}) for {typ} -- implausible unless "
                f"double leaked into the computation (double-casting-bug signature)."
            )
    return violations


def check_drift(matrix_name, typ, values, baseline):
    warnings = []
    base = baseline.get(matrix_name, {}).get(typ)
    if base is None:
        return warnings  # no baseline yet for this matrix/type -- nothing to compare
    for variant in ("plain_ferr", "quire_ferr"):
        cur = values[variant]
        b = base.get(variant)
        if not b or b == 0:
            continue
        ratio = cur / b
        if ratio > DRIFT_TOLERANCE or ratio < 1.0 / DRIFT_TOLERANCE:
            warnings.append(
                f"{matrix_name} {typ} {variant}: {cur:.3e} vs. baseline {b:.3e} "
                f"(ratio {ratio:.2f}x, tolerance ±{DRIFT_TOLERANCE}x) -- needs human review, "
                f"not auto-failed (see rajat17 for why)."
            )
    return warnings


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return os.environ.get("GIT_SHA", "unknown")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+", required=True, help="klu_quire_IR_study sweep log files to check")
    ap.add_argument("--baseline", default="ci/baseline.json", help="human-approved baseline JSON")
    ap.add_argument("--history", default="ci/regression_history.csv", help="append-only, tagged results history")
    args = ap.parse_args()

    baseline = {}
    if os.path.exists(args.baseline):
        with open(args.baseline) as f:
            baseline = json.load(f)

    sha = git_sha()
    ts = datetime.now(timezone.utc).isoformat()

    violations, drift_warnings, expected_fails, unexpected_fails = [], [], [], []
    history_rows = []

    for log_path in args.logs:
        matrix = matrix_name_from_log_path(log_path)
        with open(log_path) as f:
            text = f.read()

        fail_status = check_load_failure(text, matrix)
        if fail_status == "EXPECTED_FAIL":
            expected_fails.append(matrix)
            continue
        elif fail_status == "UNEXPECTED_FAIL":
            unexpected_fails.append(matrix)
            continue

        table = parse_working_precision_table(text)
        if not table:
            unexpected_fails.append(f"{matrix} (no Table 2 output found -- app may have crashed mid-run)")
            continue

        for typ, values in table.items():
            violations.extend(check_structural_invariant(matrix, typ, values))
            drift_warnings.extend(check_drift(matrix, typ, values, baseline))
            history_rows.append({
                "timestamp": ts, "git_sha": sha, "matrix": matrix, "type": typ,
                **values,
            })

    # Append to the tagged history file (separate from the research csv/ dir on purpose).
    os.makedirs(os.path.dirname(args.history) or ".", exist_ok=True)
    write_header = not os.path.exists(args.history)
    if history_rows:
        with open(args.history, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
            if write_header:
                w.writeheader()
            w.writerows(history_rows)

    print(f"=== Regression check @ {ts} (git {sha}) ===")
    print(f"Checked {len(args.logs)} logs: {len(history_rows)} type-rows, "
          f"{len(expected_fails)} expected failures, {len(unexpected_fails)} unexpected failures.")

    if unexpected_fails:
        print("\nUNEXPECTED FAILURES (a matrix that should load no longer does):")
        for m in unexpected_fails:
            print(f"  - {m}")

    if violations:
        print("\nSTRUCTURAL INVARIANT VIOLATIONS (hard failure):")
        for v in violations:
            print(f"  - {v}")

    if drift_warnings:
        print("\nDRIFT WARNINGS (soft -- needs human review, does not fail the job):")
        for w in drift_warnings:
            print(f"  - {w}")

    if not violations and not unexpected_fails and not drift_warnings:
        print("\nAll clear.")

    return 1 if (violations or unexpected_fails) else 0


if __name__ == "__main__":
    sys.exit(main())
