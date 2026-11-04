# Research presentation

Editable Quarto / reveal.js slides for a roughly 10-minute research meeting. The layout and sections draw on the supplied meeting example.

## Build

Install [Quarto](https://quarto.org/) and [uv](https://docs.astral.sh/uv/), then run from the repository root:

```sh
uv sync --project presentation
uv run --project presentation python presentation/make_figures.py
quarto render presentation/research-plan.qmd
```

Open `presentation/research-plan.html` in a browser; press `S` for speaker notes. Math rendering uses the MathJax CDN, and the Muon algorithm image comes from Keller Jordan’s original post; both require an internet connection.

The loss figure is a **synthetic illustration**, not a measured training run. Its random seed and plotting code are in `make_figures.py`. The second figure illustrates Bregman divergence as the gap above a tangent. No neural network training or optimizer benchmark is performed. The virtual environment is local and can be recreated from `uv.lock`.
