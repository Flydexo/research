"""Generate deterministic illustrations for the presentation; no training is run."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.labelcolor": "#35455c",
                     "xtick.color": "#58687c", "ytick.color": "#58687c",
                     "svg.fonttype": "none", "figure.facecolor": "white"})
BLUE = "#33398d"
TEAL = "#188e99"

# An invented curve, not a measured experiment or an optimizer comparison.
rng = np.random.default_rng(23)
steps = np.arange(240)
trend = 1.9 * np.exp(-steps / 54) + 0.23
noise = rng.normal(size=steps.size) * (0.10 * np.exp(-steps / 160) + 0.012)
loss = np.maximum(trend + noise, 0.12)
fig, ax = plt.subplots(figsize=(6.2, 4.1), layout="constrained")
ax.plot(steps, loss, color=BLUE, lw=1.6, alpha=.65)
ax.plot(steps, trend, color=TEAL, lw=2.5, label="Underlying trend")
ax.set(xlabel="Training step", ylabel="Loss", title="Example training loss")

ax.grid(axis="y", color="#e6e9f1")
ax.set_xlim(0, 240)
ax.set_ylim(bottom=0)
fig.savefig(OUT / "loss.svg", bbox_inches="tight")
plt.close(fig)

# Convex potential and its tangent: the vertical gap is the Bregman divergence.
x = np.linspace(-0.1, 2.4, 300)
y = .75
x0 = 1.85
psi = lambda z: .46 * z**2 + .13 * z**3
slope = .92*y + .39*y**2
tangent = psi(y) + slope * (x-y)
fig, ax = plt.subplots(figsize=(6.2, 4.2), layout="constrained")
ax.plot(x, psi(x), color=BLUE, lw=2.7, label=r"Potential $\psi$")
ax.plot(x, tangent, color=TEAL, lw=2.0, ls="--", label="Tangent at y")
ax.plot([x0,x0],[psi(y)+slope*(x0-y),psi(x0)], color="#d16c48", lw=3)
ax.scatter([y,x0],[psi(y),psi(x0)], color=BLUE, s=30, zorder=3)
ax.annotate(r"$D_\psi(x,y)$", (x0, (psi(x0)+psi(y)+slope*(x0-y))/2), xytext=(8,0), textcoords="offset points", va="center", color="#ac4d2e", fontsize=12)
ax.set_xticks([y,x0], ["y", "x"])
ax.set_yticks([])
ax.set(xlabel="Parameter", ylabel="Potential", title="A divergence as a tangent gap")
ax.legend(frameon=False, loc="upper left", fontsize=10)
ax.set_xlim(-.1,2.4)
ax.set_ylim(-.15,4.0)
fig.savefig(OUT / "bregman.svg", bbox_inches="tight")
plt.close(fig)

# Matplotlib's SVG path data can contain trailing whitespace; keep files git-clean.
for name in ("loss.svg", "bregman.svg"):
    path = OUT / name
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
