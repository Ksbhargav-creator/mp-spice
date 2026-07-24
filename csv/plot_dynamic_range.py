"""Plot the normalized dynamic-range histogram -- one grouped bar chart per
(Matrix, Experiment, Type), comparing each Variant's (Plain vs Quire, or
whatever variants exist) normalized distribution side by side.

Run AFTER normalize_dynamic_range.py has produced dynamic_range_normalized.csv,
from inside the csv/ directory:

    python3 plot_dynamic_range.py

Writes one PNG per (Matrix, Experiment, Type) group, e.g.
dynamic_range_rajat30_WorkingPrecisionResidualIR_posit32_2.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("dynamic_range_normalized.csv")

for (matrix, experiment, type_), g in df.groupby(["Matrix", "Experiment", "Type"]):
    variants = sorted(g["Variant"].unique())
    decades = sorted(g["RangeDecades"].unique())

    x = np.arange(len(decades))
    width = 0.8 / max(len(variants), 1)

    plt.figure(figsize=(8, 5))
    for i, v in enumerate(variants):
        sub = (
            g[g["Variant"] == v]
            .set_index("RangeDecades")
            .reindex(decades, fill_value=0)
        )
        plt.bar(x + i * width, sub["Percentage"], width, label=v)

    plt.xticks(x + width * (len(variants) - 1) / 2, decades)
    plt.xlabel("Dynamic range (decades)")
    plt.ylabel("% of dot products")
    plt.title(f"{matrix} — {experiment} — {type_}")
    plt.legend()
    plt.tight_layout()

    safe_type = type_.replace("<", "").replace(">", "").replace(",", "_")
    out = f"dynamic_range_{matrix}_{experiment}_{safe_type}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")
