#!/usr/bin/env python3
"""Regenerate the README's result tables from klu_quire_IR_study logs, and
diff the regenerated Table 2 against the numbers currently in README.md.

    make_readme_tables.py --logs-dir results/logs --csv-dir results/csv \
        --readme README.md [--missing "rajat03"] [--failed "..."]

Prints Markdown to stdout. Nothing here re-derives a number: every value is
parsed straight from the logs the solver wrote, so a mismatch against the README
is a real difference in the run, not an artifact of this script.

Tables produced:
  * Table 1 -- factor in posit, refine with a double-precision residual
               (posit<16,2>/posit<32,2>, plain vs quire factorization accumulator)
  * Table 2 -- working-precision residual (posit<32,2>/posit<64,3>, plain vs quire
               residual accumulator), plus the quire-improvement factor
  * Dynamic range (mean/median/max decades per dot product, posit<32,2>, Plain)
  * Reproduction vs README: Table 2 forward errors compared row by row

NOT reproducible from a current run (and so deliberately absent): the README's
Forward/Backward/Schur accumulation-length tables -- that instrumentation was
stripped from the app and is documented there as historical.
"""
import argparse
import glob
import os
import re
import sys

REL_TOL = 0.05   # README-vs-rerun forward error agreement threshold (relative)


def parse_cell(tokens):
    """tokens = ['res', 'ferr', 'it'] as strings, or ['FAIL', '-', '-']."""
    if not tokens or tokens[0] == "FAIL":
        return None
    try:
        return {"res": tokens[0], "ferr": tokens[1], "it": tokens[2]}
    except IndexError:
        return None


def parse_table(text, header_substring):
    """Rows of the table whose title line contains header_substring.
    -> {type: {"plain": cell|None, "quire": cell|None}}"""
    rows, in_table = {}, False
    for line in text.splitlines():
        if not in_table:
            if header_substring in line:
                in_table = True
            continue
        if not line.strip():
            if rows:
                break
            continue
        if "|" not in line or not line.lstrip().startswith("posit<"):
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        typ = parts[0].strip()
        rows[typ] = {"plain": parse_cell(parts[1].split()),
                     "quire": parse_cell(parts[2].split())}
    return rows


def parse_ieee(text, header_substring):
    """IEEE baseline rows ('float' / 'double') under a title containing header_substring.
    -> {type: {"plain": cell|None, "quire": cell|None}}  ('quire' = wide accumulator)"""
    rows, in_table = {}, False
    for line in text.splitlines():
        if not in_table:
            if header_substring in line:
                in_table = True
            continue
        if not line.strip():
            if rows:
                break
            continue
        if "|" not in line or line.split("|")[0].strip() not in ("float", "double"):
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        rows[parts[0].strip()] = {"plain": parse_cell(parts[1].split()),
                                  "quire": parse_cell(parts[2].split())}
    return rows


def ieee_table(data, key, title):
    lines = [f"### {title}", "",
             "| Matrix | n | Type | res (plain) | ferr (plain) | it | res (wide acc) | ferr (wide acc) | it |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m, d in sorted(data.items()):
        for t in ("float", "double"):
            r = d.get(key, {}).get(t)
            if r:
                lines.append(f"| {m} | {d['n']} | {t} | {cell_str(r['plain'])} | {cell_str(r['quire'])} |")
    return lines


def matrix_n(text):
    m = re.search(r"Matrix:\s*(\d+)\s*x\s*(\d+)", text)
    return int(m.group(1)) if m else None


def load_logs(logs_dir):
    out = {}
    for path in sorted(glob.glob(os.path.join(logs_dir, "*_sweep.log"))):
        name = os.path.basename(path)[: -len("_sweep.log")]
        with open(path, errors="replace") as f:
            text = f.read()
        t1 = parse_table(text, "Native KLU + double-residual")
        t2 = parse_table(text, "Working-precision residual")
        i1 = parse_ieee(text, "IEEE baseline: native KLU")
        i2 = parse_ieee(text, "IEEE baseline: working-precision")
        if t1 or t2:
            out[name] = {"n": matrix_n(text), "t1": t1, "t2": t2, "i1": i1, "i2": i2}
    return out


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def cell_str(cell):
    return f"{cell['res']} | {cell['ferr']} | {cell['it']}" if cell else "FAIL | - | -"


def table1(data, types=("posit<16,2>", "posit<32,2>")):
    lines = ["### Table 1 -- factor in posit, refine with double residual", "",
             "| Matrix | n | Type | res (plain) | ferr (plain) | it | res (quire) | ferr (quire) | it |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m, d in sorted(data.items()):
        for t in types:
            r = d["t1"].get(t)
            if not r:
                continue
            lines.append(f"| {m} | {d['n']} | {t} | {cell_str(r['plain'])} | {cell_str(r['quire'])} |")
    return lines


def improvement(plain, quire):
    p, q = fnum(plain["ferr"]) if plain else None, fnum(quire["ferr"]) if quire else None
    if not p or not q:
        return None
    return p / q


def table2(data, typ):
    lines = [f"### Table 2 -- working-precision residual, {typ}", "",
             "| Matrix | n | Forward Error (plain) | Iterations | Forward Error (quire) | Iterations | Quire improvement |",
             "|---|---|---|---|---|---|---|"]
    rows = []
    for m, d in data.items():
        r = d["t2"].get(typ)
        if r and r["plain"] and r["quire"]:
            rows.append((fnum(r["plain"]["ferr"]) or 0.0, m, d["n"], r))
    for _, m, n, r in sorted(rows, reverse=True):   # worst-to-best by plain forward error, as in the README
        imp = improvement(r["plain"], r["quire"])
        imp_s = "n/a" if imp is None else (f"{imp:.1f}x" if imp >= 1 else f"{imp:.2f}x (quire worse)")
        lines.append(f"| {m} | {n} | {r['plain']['ferr']} | {r['plain']['it']} | "
                     f"{r['quire']['ferr']} | {r['quire']['it']} | {imp_s} |")
    return lines


def dynamic_range(csv_dir):
    path = os.path.join(csv_dir, "dynamic_range_summary.csv")
    if not os.path.exists(path):
        return ["### Dynamic range", "", "(dynamic_range_summary.csv not found -- normalize step did not run)"]
    import pandas as pd
    df = pd.read_csv(path)
    df = df[(df.Experiment == "WorkingPrecisionResidualIR") & (df.Type == "posit<32,2>")
            & (df.Variant == "Plain")].sort_values("Matrix")
    lines = ["### Dynamic range per dot product (posit<32,2>, Plain)", "",
             "| Matrix | n | Mean | Median | P90 | Max |", "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r.Matrix} | {int(r.n)} | {r.Mean:.2f} | {r.Median:g} | {r.P90:g} | {int(r.Max)} |")
    return lines


def direct_solve(logs_dir):
    """Verbatim excerpt of klu_quire_study's table (README 'Quire in the KLU
    factorization', direct solve, no refinement). Not diffed automatically: the
    README keeps only 3 significant figures for it."""
    lines = ["### Direct solve, plain vs quire factorization accumulator (klu_quire_study)", ""]
    paths = sorted(glob.glob(os.path.join(logs_dir, "*.direct.log")))
    if not paths:
        return lines + ["(no klu_quire_study output found)"]
    for p in paths:
        name = os.path.basename(p)[: -len(".direct.log")]
        with open(p, errors="replace") as f:
            text = f.read().splitlines()
        block, on = [], False
        for ln in text:
            if "Direct sparse-LU" in ln:
                on = True
            if on:
                if not ln.strip() and block:
                    break
                block.append(ln)
        lines += [f"**{name}**", "", "```", *(block or ["(table not found in log)"]), "```", ""]
    return lines


README_T2_ROW = re.compile(
    r"^\|\s*((?:rajat|add)\d+)\s*\|\s*(\d+)\s*\|\s*([\d.eE+-]+)\s*\|\s*(\d+)\s*\|\s*"
    r"([\d.eE+-]+)\s*\|\s*(\d+)\s*\|")


def readme_table2(readme_path):
    rows = {}
    with open(readme_path, errors="replace") as f:
        for line in f:
            m = README_T2_ROW.match(line)
            if m:
                name, _n, pf, pit, qf, qit = m.groups()
                rows[name] = {"plain_ferr": float(pf), "plain_it": int(pit),
                              "quire_ferr": float(qf), "quire_it": int(qit)}
    return rows


def compare_readme(data, readme_path):
    lines = ["### Reproduction vs README (Table 2, posit<32,2>)", ""]
    if not readme_path or not os.path.exists(readme_path):
        return lines + ["(README not available to compare against)"]
    ref = readme_table2(readme_path)
    lines += [f"Forward errors compared with relative tolerance {REL_TOL:.0%}; iteration counts compared exactly.", "",
              "| Matrix | plain ferr (README -> rerun) | quire ferr (README -> rerun) | iterations (README -> rerun) | verdict |",
              "|---|---|---|---|---|"]
    n_match = n_diff = 0
    for m in sorted(ref):
        r = data.get(m, {}).get("t2", {}).get("posit<32,2>")
        if not (r and r["plain"] and r["quire"]):
            lines.append(f"| {m} | {ref[m]['plain_ferr']:.3e} -> (no result) | {ref[m]['quire_ferr']:.3e} -> (no result) | - | NOT RERUN |")
            continue
        pf, qf = fnum(r["plain"]["ferr"]), fnum(r["quire"]["ferr"])
        ok_p = abs(pf - ref[m]["plain_ferr"]) <= REL_TOL * abs(ref[m]["plain_ferr"])
        ok_q = abs(qf - ref[m]["quire_ferr"]) <= REL_TOL * abs(ref[m]["quire_ferr"])
        ok_it = (int(r["plain"]["it"]) == ref[m]["plain_it"]) and (int(r["quire"]["it"]) == ref[m]["quire_it"])
        verdict = "match" if (ok_p and ok_q and ok_it) else "DIFFERS"
        n_match += verdict == "match"; n_diff += verdict == "DIFFERS"
        lines.append(f"| {m} | {ref[m]['plain_ferr']:.3e} -> {pf:.3e} | {ref[m]['quire_ferr']:.3e} -> {qf:.3e} | "
                     f"{ref[m]['plain_it']}/{ref[m]['quire_it']} -> {r['plain']['it']}/{r['quire']['it']} | {verdict} |")
    lines += ["", f"**{n_match} match, {n_diff} differ** (of {len(ref)} README rows).",
              "A difference is a finding, not necessarily a bug: the README numbers came from a macOS "
              "(Apple Clang) build, this ran under Linux (GCC); check the toolchain and the Universal/MTL5 "
              "revisions in PROVENANCE.txt before concluding anything."]
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-dir", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--readme", default=None)
    ap.add_argument("--missing", default="", help="matrices that never finished (space separated)")
    ap.add_argument("--failed", default="", help="matrices whose run exited nonzero (space separated)")
    args = ap.parse_args()

    data = load_logs(args.logs_dir)
    out = ["# Regenerated results", "",
           f"Parsed {len(data)} logs from `{args.logs_dir}`.", ""]
    if args.failed.strip():
        out += [f"**Failed runs (nonzero exit):** {args.failed.strip()}", ""]
    if args.missing.strip():
        out += [f"**Never finished (no status file -- killed, timed out, or not scheduled):** {args.missing.strip()}", ""]
    out += table1(data) + [""]
    out += table2(data, "posit<32,2>") + [""]
    out += table2(data, "posit<64,3>") + [""]
    out += ieee_table(data, "i1", "Table 3a -- IEEE baseline: factor in float/double, double residual") + [""]
    out += ieee_table(data, "i2", "Table 3b -- IEEE baseline: working-precision residual, plain vs wide accumulator") + [""]
    out += dynamic_range(args.csv_dir) + [""]
    out += direct_solve(args.logs_dir) + [""]
    out += compare_readme(data, args.readme)
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
