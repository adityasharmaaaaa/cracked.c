import csv
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "bench/results/matmul_baseline.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "bench/results/matmul_baseline.png"

# series[(impl, label)] = [(size, best_gflops), ...]
series = defaultdict(list)
with open(src, newline="") as f:
    for row in csv.DictReader(f):
        label = ("T" if row["transpose_a"] == "1" else "N") + ("T" if row["transpose_b"] == "1" else "N")
        series[(row["impl"], label)].append((int(row["size"]), float(row["gflops_best"])))

fig, ax = plt.subplots(figsize=(8, 5))
for (impl, label), pts in sorted(series.items()):
    pts.sort()
    style = "-" if impl == "naive_ikj" else "--"
    ax.plot([p[0] for p in pts], [p[1] for p in pts], style, marker="o", label=f"{impl} {label}")

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("matrix size N (N x N x N matmul)")
ax.set_ylabel("GFLOPS (best of repetitions)")
ax.set_title("cracked.c: naive matmul baseline")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(dst, dpi=160)
print("wrote", dst)