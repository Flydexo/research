"""Plot singular-value bars from the Day 5 polar diagnostic."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).parent
RESULTS = json.loads((HERE / "day5-polar-diagnostic-results.json").read_text())
CASES = {x["case"]: x for x in RESULTS["cases"] if x["ns_dtype"] == "torch.bfloat16"}

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True, constrained_layout=True)
colors = ["#3B6EA8", "#269578", "#D47B2C"]
stages = ["Input gradient $G$", "Exact partial polar $O$", "Five-step NS $N$"]

for ax, key, title in zip(
    axes,
    ["tall_full_rank", "tall_rank_one"],
    ["$3\\times2$ full rank", "$3\\times2$ rank one"],
):
    case = CASES[key]
    groups = [
        case["input_singular_values"],
        case["exact_singular_values"],
        case["ns_singular_values"],
    ]
    x = np.arange(len(groups))
    width = 0.30
    for j in range(2):
        values = [group[j] for group in groups]
        bars = ax.bar(x + (j - 0.5) * width, values, width, color=colors, alpha=0.95 if j == 0 else 0.55)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.045, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(title)
    ax.set_xticks(x, stages, rotation=16, ha="right")
    ax.set_ylim(0, 3.55)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.text(0.97, 0.94, "dark: $\\sigma_1$\nlight: $\\sigma_2$", transform=ax.transAxes, ha="right", va="top", fontsize=9)

axes[0].set_ylabel("Singular value (bar height)")
fig.suptitle("Polar update: exact SVD versus finite Newton–Schulz", fontsize=14)
fig.savefig(HERE / "day5-spectral-illustration.png", dpi=180, bbox_inches="tight")
