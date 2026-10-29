# One-week curriculum: non-Euclidean optimization for transformers

Prepared 25 October 2026. Workload: 4–6 focused hours/day. Assumes practical PyTorch and LLM training experience; mathematical familiarity is checked on Day 1. Objective: prepare a 10-minute presentation and 15-minute discussion, while building foundations for original optimizer research.

## The question that organizes the week

**How should we turn a stochastic derivative into a parameter update, given the geometry of a neural network?**

Your project brief includes norms, Bregman divergences, Frank–Wolfe, theory, and experiments. For this week, follow:

**Local models → norms and duality → mirror descent / Frank–Wolfe → matrix geometry → Muon / Scion → controlled experiments.**

A realistic outcome is the ability to derive the core updates, explain their assumptions, and formulate testable hypotheses. Mastering every convergence proof or producing a new record takes longer.

Non-Euclidean does not necessarily mean curved manifolds or hyperbolic geometry. Changing the norm on a vector space already qualifies. Euclidean gradient descent is mathematically valid: it makes an implicit geometric choice. The question is whether that choice is useful.

## Daily routine

Four-hour default: 20 min closed-book recall; 70 min targeted reading; 70 min derivation; 60 min implementation, paper audit, or presentation work; 20 min spoken explanation and questions. Reading-heavy days use the specific allocation below instead. Time estimates are caps for a focused first pass, not promises of mastery. Breaks are additional to focused hours.

An **extension** is an optional deeper exercise or reading, not a prerequisite for finishing the day. Use an optional fifth/sixth hour for it. If behind, drop extensions and training runs before dropping derivations or rehearsal. For every paper record: problem, update, geometry, assumptions, evidence, limitation, and one question. Mark each claim as theorem, empirical observation, or hypothesis.

## Day 1 — What is the optimization problem, and what does geometry change?

**Read in this order (replacement for SGD.pdf):**

1. [Cornell CS4787, Lecture 3 (2021)](https://www.cs.cornell.edu/courses/cs4787/2021sp/notebooks/Slides3.html): focus on smoothness and the descent inequality; skip familiar ML setup (35–45 min).
2. [Cornell, Lecture 4 (2021)](https://www.cs.cornell.edu/courses/cs4787/2021sp/notebooks/Slides4.html): step size, stationarity, strong convexity, and conditioning (35–45 min).
3. [Boyd & Vandenberghe, Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf): §9.4 opening and §9.4.1, starting at printed p. 475; focus on Euclidean and quadratic norms (30–40 min). The geometric construction is broadly useful; the chapter's convex convergence assumptions are not automatically valid for neural networks.
4. [Cornell, Lecture 6 (2025)](https://www.cs.cornell.edu/courses/cs4787/2025fa/lectures/lecture6.html): step size, minibatches, and gradient noise (25–35 min).

**Why this change:** this sequence makes assumptions and the geometry explicit. Keep local `SGD.pdf` only as an optional intuitive supplement. Its printed p. 13 overstates convergence to a local minimum; descent and stationarity are different claims. Move momentum to Day 2.

**Four-hour allocation:** 20 min recall/prerequisite check, 125 min reading at the lower time estimates, 50 min derivation, 30 min quadratic diagnostic, 15 min oral check. With 5–6 hours, expand the derivation and diagnostic rather than adding more sources.

**Learn:** empirical versus population risk; minibatch gradient noise; smoothness versus convexity; conditioning; training objective versus validation criterion. Write the language-model objective as an expectation of next-token negative log-likelihood. Distinguish reducing training loss from reaching a validation target and from generalizing beyond that benchmark.

**Derive:** minimizing <g, Δ> + ||Δ||₂²/(2η) gives Δ = −ηg. Replacing the quadratic by ΔᵀHΔ/(2η), H positive definite, gives Δ = −ηH⁻¹g. H is a chosen metric/preconditioner, not automatically the Hessian.

Derive the descent lemma and show that an L-smooth function with η ≤ 1/L satisfies f(x−ηg) ≤ f(x)−η||g||₂²/2 for the exact gradient. Identify what changes with stochastic gradients. Convexity is not needed for this one-step statement.

**Diagnostic exercise:** use f(x)=½xᵀAx with A=diag(1,100), then rotate A. Plot gradient descent and diagonal preconditioning trajectories. Reuse the same starting point and show stable and unstable learning rates. The rotation exposes the limitations of coordinatewise adaptation.

**Exit test:** explain why a gradient specifies first-order sensitivity but selecting the steepest direction requires a notion of unit step. Explain why stationarity is weaker than global optimality.

**Deliverable:** one page defining the research problem and one annotated quadratic plot. Extension: diagonalize A and derive the stable interval 0<η<2/λmax(A).

## Day 2 — Adaptive baselines and dual norms

**Read in order:**

1. [Cornell, Lecture 8 (2025): Preconditioning and Adaptive Learning Rates](https://www.cs.cornell.edu/courses/cs4787/2025fa/lectures/lecture8.html), preconditioning → AdaGrad/RMSProp → Adam (25 min). This supplies the geometric motivation before the original papers. If momentum is unclear, use [Lecture 7](https://www.cs.cornell.edu/courses/cs4787/2025fa/lectures/lecture7.html) for a 15-minute refresher.
2. Local `Adam.pdf`, Algorithm 1 and bias-correction discussion (10 min); `AdamW.pdf`, §2 and its algorithm (15 min). Retain the original papers as the authoritative definitions; there is no need to replace them with another tutorial.
3. Local `modular-duality-in-DL.pdf`, §§3.1–3.3 (20 min), using yesterday's Boyd section as support. Work through the vector examples today; derive the matrix rows of the table on Day 5.

**Stop after:** you can reconstruct Adam/AdamW and the vector duality examples. Defer the original Adam convergence proof and lengthy empirical comparisons.

**Learn:** momentum, second moments, bias correction, epsilon, diagonal preconditioning, decoupled decay. Adam's second moment is an uncentered gradient statistic, not a Hessian estimate. Its historical convergence claims require care; failures under some settings do not imply it is unusable in practice.

**Derive:** for a norm ||·||, define ||g||* = max_{||s||≤1}<g,s>. Let d(g) be any maximizer. For g≠0 and L>0, one solution of

    min_Δ <g,Δ> + (L/2)||Δ||²

is Δ = −(||g||*/L)d(g). Separate direction from magnitude: −d(g) alone is a normalized step, not the same update.

Build this table yourself, then check it:

| Unit ball for updates | Dual norm of gradient | A minimizing unit-ball direction |
|---|---|---|
| ℓ₂ | ℓ₂ | −g/||g||₂ |
| ℓ∞ | ℓ₁ | −sign(g) |
| ℓ₁ | ℓ∞ | −sign(gⱼ)eⱼ, j maximizing abs(gⱼ) |
| Frobenius matrix norm | Frobenius | −G/||G||F |
| Spectral matrix norm | Nuclear | −UᵣVᵣᵀ |
| Nuclear matrix norm | Spectral | −u₁v₁ᵀ |

Use the Frobenius inner product for matrices; Uᵣ,Vᵣ contain nonzero singular directions. Zero gradients admit a zero update; optimizers can be nonunique.

**Exercise:** derive why coupled L2 gives −ηP(g+λw), while decoupled decay gives −ηPg−ηλw for a fixed preconditioner P. Actual Adam also incorporates coupled regularization into its moment history. Show why the two procedures differ.

**Exit test:** derive the three vector directions without notes; explain why a spectral-ball LMO generally produces a many-rank update, while a nuclear-ball LMO can be rank one.

**Deliverable:** an update-rule comparison sheet. Extension: read the abstract and counterexample discussion of Reddi et al., *On the Convergence of Adam and Beyond*: https://openreview.net/forum?id=ryQu7f-RZ.

## Day 3 — Bregman divergences and mirror descent

**Read in order:** retain Mark Reid's *Meet the Bregman Divergences* (20 min), then Bubeck's *Convex Optimization: Algorithms and Complexity*, specifically §§4.1–4.3: Mirror maps, Mirror descent, and Standard setups (50 min). This remains a strong pairing: intuition followed by a precise algorithm and examples. On the first pass, prioritize definitions, the update, and Euclidean/entropy examples; move the convergence proof to an extension. Skip Mirror Prox and dual averaging. [Searchable full text](https://arxiv.org/html/1405.4980v2).

- https://mark.reid.name/blog/meet-the-bregman-divergences.html
- https://arxiv.org/abs/1405.4980

**Derive:** for a differentiable strictly convex potential h,

    D_h(x,y) = h(x) − h(y) − <∇h(y),x−y>.
    x_next = argmin_{x∈C} {η<g,x> + D_h(x,x_current)}.

Show that h(x)=½||x||₂² recovers projected gradient descent. On the probability simplex, h(x)=Σxᵢlog xᵢ gives KL divergence and, from positive coordinates,

    x_next,i ∝ x_i exp(−ηg_i).

Under appropriate interior/domain conditions, the unconstrained optimality condition is ∇h(x_next)=∇h(x)−ηg. Explain the domain restrictions rather than assuming ∇h is globally invertible.

**Exercise:** compare entropy mirror descent and Euclidean projected descent on a small simplex-constrained quadratic. Verify positivity and sum-to-one; explain the role of initialization.

**Exit test:** why is a Bregman divergence generally not a metric? What is the potential h, and how is it different from the training loss? How does this differ from merely changing a fixed norm? Natural gradient uses a statistical metric, often the Fisher; related geometry does not make every method identical.

**Deliverable:** one diagram explaining tangent approximation error and one mirror-descent derivation. Extension: derive the three-point Bregman identity; defer the full convergence proof.

## Day 4 — Frank–Wolfe and the constrained/unconstrained distinction

**Read in order:** local `frank-wolfe.pdf`, slides 3–6 and one example (10 min); then [Jaggi, Revisiting Frank–Wolfe (2013)](https://proceedings.mlr.press/v28/jaggi13.html), Algorithm 1, §2 (duality gap), and the curvature/convergence statements in §3 (30 min). Finally read local `training-DLM-norm-constrained-LMO.pdf`, Algorithms 1–2 and the surrounding explanation in §§2–3 (30 min).

**Why this change:** the slides remain a quick introduction; Jaggi is the primary reference for certificates and precise convex guarantees. Read its theorem statements first, not the entire proof or application catalogue. Scion then supplies the distinct unconstrained construction. Appendix A.1 of Scion is optional follow-up.

**Derive and compare:**

    s_t ∈ argmin_{s∈C}<g_t,s>
    Frank–Wolfe: x_next = (1−γ)x + γs_t

For convex C, feasible x and s_t, and γ∈[0,1], feasibility is preserved. Compare the Euclidean projection problem with this linear minimization problem: projection-free does not mean computation-free.

Now let s_t minimize <m_t,s> over an update norm-ball:

    unconstrained normalized update: x_next = x + γs_t

The same oracle can appear in both formulas, but the second does not keep weights inside the oracle's ball. Distinguish a constraint on parameters from a bound on each update.

**Proof exercise:** define the deterministic Frank–Wolfe gap G_FW(x)=max_{s∈C}<∇f(x),x−s>. Show it upper-bounds suboptimality for convex f; in nonconvex problems it measures constrained first-order stationarity. Prove a one-step descent inequality and understand where compactness, smoothness, and convexity enter an O(1/t) convex result.

**Exit test:** why can a constrained optimum have nonzero gradient? Why is momentum useful with noisy LMOs? Why doesn't an unconstrained LMO update inherit all classical Frank–Wolfe guarantees?

**Deliverable:** a page comparing projected GD, mirror descent, Frank–Wolfe, and unconstrained LMO updates. Extension: implement ℓ₁-ball Frank–Wolfe on a small least-squares problem and inspect sparsity.

## Day 5 — Matrix geometry, Muon, and modular norms

**Bridge from Day 4: spectral descent (30–45 min, prioritized extension).** Before the Muon reading, study [Carlson et al., Preconditioned Spectral Descent for Deep Learning (2015)](https://papers.nips.cc/paper/2015/file/f50a6c02a3fc5a3a5d4d9391f05f3efc-Paper.pdf): §2.1 carefully (15 min), then §2.2 and Figure 1 (10–15 min); skim §§2.3–2.4 only if time remains. In this literature SSD means **Stochastic Spectral Descent**. The earlier RBM-specific paper is historical background, not an additional assignment.

**Purpose and exercise:** connect Day 4's spectral-ball LMO to norm-based local models. For G = UΣVᵀ, compare minimizing <G,Δ> subject to ||Δ||op ≤ 1 with minimizing <G,Δ> + (L/2)||Δ||op² for L > 0. Derive both using the nuclear dual norm, and explain the shared direction and different magnitudes. Fold this exercise into the existing matrix-direction derivation below. When reading Muon, identify what momentum, approximate polar computation, and update scaling change. Treat the 2015 experiments as evidence in their own model settings, not modern transformer benchmarks.

**Scheduling:** use the first 30–45 minutes of the optional fifth hour for this bridge, ahead of Shampoo and the optional overview. If limited to four hours, skim §2.1 for 10–15 minutes within the existing derivation block and defer §§2.2–2.4. Preserve the current modular-duality and Moonshot core readings.

**Read in order:**

1. [Keller Jordan's Muon post](https://kellerjordan.github.io/posts/muon/): definition, design, and Newton–Schulz code (20 min). Keep this as the practical introduction; its dated results are historical evidence, not a current leaderboard.
2. Local `modular-duality-in-DL.pdf`, §§3.3–3.4 and the linear-layer computational discussion (50 min). Revisit the matrix duality map in §3.3; in §3.4 identify the module attributes, the two well-normed inequalities, and how composition combines module norms. Work through one two-module composition, then use the linear-layer example to connect the construction to an update direction. This is the main source for deriving the direction.
3. [Moonshot AI, Muon is Scalable for LLM Training](/Users/matheoledevehat/Documents/research/muon-scalable-kimi.pdf), local arXiv v1 (24 February 2025): §2.2 and §3.1/Table 1 (20 min). Focus on decoupled weight decay, matrix-shape-dependent update RMS, and the distinction between directly normalizing an update and applying a shape-based scale. Skim §2.1 only to reconcile notation with Jordan's post.

**Why the Moonshot addition:** it connects the idealized matrix update to practical large-model training choices. It replaces the overview reading in this day's core. With §3.4 added, the Day 5 core reading totals 90 minutes; the additional 20 minutes come from the implementation block as noted below. [Pethick et al., Training Neural Networks at Any Scale](https://arxiv.org/html/2511.11163v1) becomes an optional 20-minute synthesis. Distributed implementation details are optional after the week unless systems questions become central.

**Time adjustment:** §3.4 is now part of the Day 5 core reading. Use 20 minutes from the usual implementation block for it; preserve the matrix-direction derivation and exact-versus-approximate polar comparison. Skip lengthy proof details in §3.4 on the first pass.

**What stays optional:** §§2–3 of local `scalable-opti-modular-norm.pdf` and a deeper proof or implementation of the full modular composition rules. Use these only after the core derivation and diagnostic are complete.

**Learn:** a weight matrix is a linear map between activation spaces. Derive, for W of shape d_out × d_in,

    ||W||RMS→RMS = sqrt(d_in/d_out) ||W||spectral.

Explain why embeddings and hidden linear layers can warrant different norms. Distinguish normalizing an existing update from deriving a direction through duality.

**Derive:** using G=UΣVᵀ and the trace inequality, solve the spectral-ball LMO. Understand the polar factor UᵣVᵣᵀ and what happens to singular values. Distinguish orthogonalizing an update from constraining the weights to be orthogonal. Rectangular matrices cannot have both row and column orthonormality in general.

**Exercise:** on small matrices compare an exact SVD polar update with the finite Newton–Schulz routine from the Muon post. Inspect singular values, descent alignment, approximation error, dtype, and runtime. Include rectangular and rank-deficient cases. Finite practical polynomial iterations need not return the exact polar factor; use exact SVD as an educational reference.

**Moonshot derivation (within the existing derivation block):** for an exact full-rank polar factor O of shape A × B, show ||O||F² = min(A,B), hence RMS(O) = 1/sqrt(max(A,B)). Explain the paper's scale 0.2 sqrt(max(A,B)) and why it yields RMS 0.2 in this ideal case. The 0.2 coefficient is an empirical calibration, not a universal constant. For the rank-r partial polar factor, RMS is sqrt(r/(AB)); finite Newton–Schulz outputs need not satisfy the ideal identity. Compare this scaling with the RMS→RMS operator-norm scaling above: they encode different normalization choices.

**Exit test:** explain momentum → approximate polar transformation → scaling → parameter update. Explain the original hidden-matrix/auxiliary-optimizer split. Do not assume every later Muon implementation is identical. Explain why matching update RMS does not make Muon equivalent to AdamW, and why better polar approximation need not improve training.

**Deliverable:** one spectral illustration and an annotated update. Lower-priority extension, after the SSD bridge: read Shampoo's abstract and matrix-case Algorithm 1: https://arxiv.org/abs/1802.09568. Compare accumulated matrix preconditioners with a polar transformation of the current momentum; they are not generally the same algorithm.

## Day 6 — Scion, theory audit, and your benchmark protocol

**Read:** local `training-DLM-norm-constrained-LMO.pdf`, §§3–6, Tables 1–4, and Algorithm 3. Read §5 for assumptions and conclusions first; enter a proof only after stating its claim correctly. Read the [optimization-track README and baseline script](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_3_optimization). Keep Scion as the central primary source: a broad tutorial cannot replace its precise update and theorem assumptions. Budget 45 min for its key algorithms/tables, 30 min for one theorem audit, and 25 min for the benchmark; use 40 min of the usual implementation block for this extra reading.

**Optional comparison, 20–30 min:** [SOAP: Improving and Stabilizing Shampoo using Adam](https://arxiv.org/abs/2409.11321), abstract and algorithm description. Read it after the Day 5 Shampoo extension. Its adaptive updates in a learned coordinate basis provide a useful comparison with polar updates; this is relevant to exploring stronger Track 3 optimizers. Do not import its reported improvements as predictions for your benchmark.

**Paper audit:** for each selected theorem write the objective class, smoothness norm, stochastic assumptions, schedule, output iterate, and quantity controlled. Pay attention to constant versus vanishing momentum, noise terms, and random-iterate guarantees. A stationarity result does not guarantee faster transformer training or improved validation loss. Treat model-size transfer and reported training improvements as experimental claims in their tested settings.

**Benchmark facts, checked 25 October 2026:** Track 3 minimizes steps to validation loss below 3.28 with fixed data, batch size, and architecture, and at most one forward/backward pass per step. Initialization and optimizer schedules may change. Its published acceptance condition is (3.28−mean_loss)√n ≥ 0.004 over non-cherry-picked runs. It disallows per-run validation-based stopping; a common evaluation step can be selected across trials. Reproducible submissions include the complete optimizer code in the logged script. Pin a commit and recheck the rules before running.

https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_3_optimization

**Moonshot evidence audit (15 minutes, replacing part of the broad Scion experiment survey):** read §3.2 and Figure 1a/Table 3; inspect Appendix B if you need details of AdamW tuning. Distinguish the dense-model scaling-law comparison from the separate large MoE training demonstration. The reported roughly twofold compute advantage is an empirical result in that setup, not a prediction of half as many Track 3 steps. Identify what changes across their compute-optimal models versus what your fixed-architecture benchmark holds constant. For a useful ablation, hold the polar direction fixed and compare scaling rules before attributing a gain to geometry alone.

**Your experimental design (recommendation):**

1. Start with the track baseline, then a tuned AdamW comparison and one geometry change. Add Scion as a distinct comparison when its parameter groups and scales are understood.
2. Keep initialization fixed for the first ablation even though the track permits changes. Freeze the data order and pair seeds where possible.
3. Allocate comparable tuning budgets. Specify learning-rate, momentum, decay, and schedule searches; equal learning rates are not equal treatment.
4. Separate exploratory tuning from fresh confirmatory runs. Report all runs, failures, variability, and tuning cost. Use a common preselected training horizon for the cleanest initial study.
5. Plot validation loss versus steps as the primary curve. Also record tokens, elapsed time, memory, and optimizer overhead. Passing the target test does not itself establish superiority over another optimizer; compare paired runs separately.
6. A small-model proxy can debug mechanics and suggest hypotheses. Label it a proxy, not an official Track 3 result. A real track run is optional this week and must fit available compute.

Log every experiment with **Trackio**: source revision, full configuration, seed, dataset/split, model shape, precision, hardware, optimizer and parameter groups, schedules, train/validation loss, steps/tokens/time, memory, and failures. For geometry diagnostics, log sampled layerwise gradient/update norms, update-to-weight ratios, and singular-value summaries at a manageable cadence. Save plots and raw logs.

Before renting a Vast.ai GPU, agree on a maximum credit-spending allowance. No rental or training run is required to complete this curriculum.

**Deliverable:** a reproducibility checklist and two falsifiable hypotheses, with the comparison needed to refute each.

Possible hypotheses, not established novel contributions:

- Partial singular-value flattening retains useful gradient magnitude information. Compare U diag(σᵖ)Vᵀ for a small p grid including 0 and 1, explicitly matching update scale and tuning LR. Define zero-singular-value handling. Inspect existing variants before claiming novelty.
- Matrix geometry and adaptive preconditioning are complementary. Compare the same baseline with each component alone and together. The track already contains combinations, so ask a narrower mechanistic question.
- Layer-specific scale control explains part of an apparent optimizer improvement. Change direction and scale separately; inspect which layer families benefit.

## Day 7 — Synthesis and presentation

**Source decision:** no additional paper improves this day enough to justify displacing rehearsal. Use your completed notes, original algorithm definitions, and the track rules to check slide claims.

No new broad reading. Spend 60 min on closed-book derivations, 90 min preparing six slides, 60 min on two timed rehearsals, and 30 min reviewing the weakest answers. Use the optional hours for a mock Q&A or finishing a diagnostic figure.

| Slide | Time | Content |
|---|---:|---|
| 1 | 1:00 | Research question and Track 3 objective |
| 2 | 1:30 | Why geometry matters: quadratic and local update model |
| 3 | 2:00 | Dual norms, LMO, and the spectral example |
| 4 | 2:00 | AdamW, Muon, Scion: choices, connections, differences |
| 5 | 1:30 | Limitations and your evidence: diagnostic result or clearly labeled proposed experiment |
| 6 | 2:00 | One hypothesis, controlled protocol, and next decision |

Keep mirror descent/Bregman details, one proof, and benchmark statistics in backup slides. Mention mirror descent in the main narrative as the broader alternative geometric framework; do not attempt every derivation in ten minutes.

Practice these questions with a 30-second answer, then a two-minute derivation:

- What exactly is non-Euclidean here? Does it require a manifold?
- Why does the derivative need a geometry to specify steepest descent?
- What is a dual norm? Derive the sign and spectral updates.
- How do spectral and nuclear balls differ?
- How do mirror descent, normalized descent, and Frank–Wolfe differ?
- Does orthogonalizing updates make weights orthogonal?
- What information does singular-value flattening remove?
- Why might momentum matter especially for normalized directions?
- Why does weight decay differ from L2 for Adam?
- What do the convergence theorems actually guarantee?
- What is expensive or approximate about the LMO implementation?
- How do embeddings, matrix weights, and scalar parameters differ?
- Could a gain come from tuning or initialization rather than geometry?
- Why is loss versus steps the principal plot for your track?
- What observation would refute your hypothesis?

## Reading priorities and stopping rules

**Essential:** the selected Cornell/Boyd passages; original Adam/AdamW algorithm definitions; Reid + Bubeck §§4.1–4.3; Jaggi's algorithm/gap/guarantees; Muon + modular duality; Scion and Track 3 rules. **Essential practical addition:** Moonshot §2.2/§3.1 and the short Day 6 evidence audit. **Optional overview:** the targeted passage from *Training Neural Networks at Any Scale*. **Optional:** SGD.pdf, detailed modular composition, Shampoo/SOAP, and Adam counterexamples. These replace or narrow previous readings; they are not extra full-paper assignments.

Antonio Silveti-Falls's [publication page](https://tonysf.github.io/publications/) remains a research map. After the week, use *Generalized Gradient Norm Clipping & Non-Euclidean (L0,L1)-Smoothness* and *On the Role of Batch Size in Stochastic Conditional Gradient Methods* to choose a deeper direction. They are follow-up candidates, not required readings.

Source selection was reviewed online on 25 October 2026. “Better” here means a better fit for the day's learning objective: tutorials for intuition, textbooks for foundations, original papers for exact algorithms and claims. Newer is not automatically better.

If you have only four hours/day, finish the equations and exit tests and perform one or two small diagnostics. If you have six, add matrix approximation diagnostics and a carefully controlled proxy training experiment. Preserve Day 7 for rehearsal either way.

## Local source map

- [Project brief](subject.pdf): scope and motivation.
- [Gradient descent notes](SGD.pdf): optional intuition only; replaced as the Day 1 backbone.
- [Adam](Adam.pdf) and [AdamW](AdamW.pdf): adaptive baseline and decay distinction.
- [Frank–Wolfe slides](frank-wolfe.pdf): constrained optimization and LMO introduction.
- [Modular duality](modular-duality-in-DL.pdf): norm-based directions and layer geometry.
- [Moonshot Muon scaling report](/Users/matheoledevehat/Documents/research/muon-scalable-kimi.pdf): update scaling and weight decay on Day 5; evidence audit on Day 6 (added 26 October 2026).
- [Modular norm](scalable-opti-modular-norm.pdf): architectural scaling and norm construction.
- [Norm-constrained LMOs / Scion](training-DLM-norm-constrained-LMO.pdf): bridge from foundations to the project research direction.

The local PDFs have their own versions; benchmark software is moving. Record exact versions whenever converting a reading insight into an experiment.

## Working with a tutor agent

Use [the tutor-agent prompt](tutor-agent-prompt.md) in a new chat with access to this folder, or attach both files. It defines scheduling, diagnostic testing, question answering, spaced review, and presentation rehearsal. Actual dates and study windows should be set with you; the curriculum does not assume a presentation date.
