# Day 4 — update-rule comparison

Working page. Fill one row at a time from memory, then check each formula and assumption against the sources. Use the same symbols throughout: current parameters `x_t`, gradient or gradient estimate `g_t`, feasible parameter set `C`, LMO point `s_t`, and step size `γ_t` or `η_t`.

| Method | Update rule | Where geometry enters | What is constrained or bounded? | Why the next point is feasible, if relevant |
|---|---|---|---|---|
| Projected gradient descent | `x_{t+1}=Π_C(x_t−η_t g_t)` (after source reminder) | Euclidean projection: `Π_C(z)∈argmin_{x∈C}||x−z||₂` (independent check) | Parameters `x_t∈C` | Projection returns a point in `C` |
| Mirror descent | `∇Φ(y_{t+1})=∇Φ(x_t)−η_tg_t`; `x_{t+1}∈argmin_{x∈C}D_Φ(x,y_{t+1})` | Bregman divergence from mirror potential `Φ` | Parameters `x_t∈C` | Bregman projection returns a point in `C` |
| Frank–Wolfe | `s_t∈argmin_{s∈C}⟨g_t,s⟩`; `x_{t+1}=(1−γ_t)x_t+γ_t s_t` | Linear minimization over `C` | Parameters `x_t∈C` | Convex combination of `x_t,s_t∈C` when `γ_t∈[0,1]` |
| Unconstrained LMO update | `x_{k+1}=x_k+γ_k LMO(d_k)` (user gave additive rule; indices aligned to paper) | LMO over a chosen norm ball `D` | Oracle point `s_k=LMO(d_k)` has `||s_k||≤ρ`, so `||x_{k+1}−x_k||≤γ_kρ`; parameters need not lie in `D` | No parameter-feasibility guarantee from this update alone |

## Assumptions and claims

- **Convexity of the feasible set** makes a convex-combination update feasible when `x_t,s_t∈C` and `γ_t∈[0,1]`. This was checked independently for Frank–Wolfe and later transferred to SCG.
- **Convexity of the objective** gives the tangent lower bound and hence `f(x_t)−f(x*)≤G_FW(x_t)`. The conclusion was stated independently; the proof needed help with signs and the maximum.
- **Smoothness of the objective** gives the one-step upper bound `f(x_{t+1})−f(x_t)≤−γ_t G_FW(x_t)+(L/2)γ_t²R²`. The substitution and final form were assembled during guided practice; closed-book exit recall failed (score 0). Repair next session.
- **Boundedness of `C`** gives finite diameter `R`; compactness also gives attainment of a continuous linear oracle and of the objective minimum in finite dimensions. The `2B` bound was derived independently when `||x||≤B` on `C`.
- **Classical convex Frank–Wolfe guarantees** use parameter feasibility and assumptions on the objective. Unconstrained LMO updates do not inherit those claims merely because the same oracle is used; this limitation still needs an independent check.

## One-sentence distinction

Projected GD, mirror descent, Frank–Wolfe, and constrained SCG keep the parameters feasible under their stated assumptions; unconstrained Scion uses the norm ball to bound each LMO output and therefore each update, while the parameters can leave that ball. This distinction needed a scalar counterexample to repair during the session.
