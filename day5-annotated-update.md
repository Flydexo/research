# Day 5 annotated matrix update

For a hidden-layer matrix `W` of shape `A × B`, let `G_t` be the current loss gradient, `μ` the momentum coefficient, `η_t` the learning rate, and `λ` the decoupled weight-decay coefficient. These coefficients have different jobs.

| Stage | Schematic equation | Meaning |
| --- | --- | --- |
| Momentum | `M_t = μ M_{t-1} + G_t` | Accumulate gradient history in a matrix buffer. |
| Matrix transformation | `O_t = NS_5(M_t)` | Five Newton–Schulz steps approximately transform the momentum matrix toward its polar factor. The ideal reduced-SVD reference for `M_t = U_r Σ_r V_rᵀ` is `U_r V_rᵀ`. Finite output singular values need not equal 1. |
| Shape scale | `α_{A,B} = 0.2 sqrt(max(A,B))` | Moonshot's shape-based multiplier targets update RMS near 0.2 for an exact full-rank polar factor. It does not measure and normalize each actual `O_t`. |
| Parameter step and decay | `W_t = (1 - η_t λ) W_{t-1} - η_t α_{A,B} O_t` | Apply decay directly to the weights and subtract the transformed, scaled direction. Decay is absent from the momentum buffer and Newton–Schulz input. |

The table is the simple-momentum form. Moonshot's implementation uses a Nesterov-style input to Newton–Schulz: after forming `M_t`, it feeds `μ M_t + G_t` rather than `M_t`. This changes the matrix transformed, but keeps the order **momentum → matrix transformation → scaling → parameter step**.

This is an annotated **Moonshot-scaled Muon variant**, not a claim that all Muon implementations use this exact decay or scale. Keller Jordan's original post describes Muon as orthogonalizing an SGD-momentum update and recommends AdamW for scalar/vector, embedding, and final classifier-head parameters. Moonshot likewise uses AdamW for its auxiliary parameter group. An embedding or LM head can be a matrix; its role, not just its tensor rank, matters for the optimizer split.

For a rank-`r` exact partial polar factor, `RMS(O_t) = sqrt(r/(AB))`. The shape scale gives exactly 0.2 RMS only when `r=min(A,B)` and the polar factor is exact. Direct update normalization would instead use `0.2 O_t/RMS(O_t)` when `RMS(O_t)>0`; it enforces the chosen RMS on the computed update. Neither equal update RMS nor a more accurate polar approximation alone implies equal training dynamics or better training.

Sources: [Keller Jordan, Muon](https://kellerjordan.github.io/posts/muon/); [Moonshot AI, Muon is Scalable for LLM Training](muon-scalable-kimi.pdf), §§2.2 and 3.1.
