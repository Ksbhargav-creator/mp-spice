import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("lu_accumulation.csv")

for t in ["posit<16,2>", "posit<32,2>"]:

    sub = df[df["Type"] == t]

    kernels = ["Forward","Backward","Schur"]
    matrices = sub["Matrix"].unique()

    x = np.arange(len(kernels))
    width = 0.15

    plt.figure(figsize=(8,5))

    for i,m in enumerate(matrices):
        vals = [
            sub[(sub.Matrix==m)&(sub.Kernel=="Forward")]["Median"].iloc[0],
            sub[(sub.Matrix==m)&(sub.Kernel=="Backward")]["Median"].iloc[0],
            sub[(sub.Matrix==m)&(sub.Kernel=="Schur")]["Median"].iloc[0]
        ]

        plt.bar(x+i*width, vals, width, label=m)

    plt.xticks(x+2*width,kernels)
    plt.ylabel("Median Accumulation Length")
    plt.title(f"Sparse LU Accumulation Depth ({t})")
    plt.legend()
    plt.tight_layout()
    plt.show()