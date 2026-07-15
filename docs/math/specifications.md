# Mathematical Specifications

This document gives formal definitions of the KPIs and cost models used
in CHEN. All formulas here are implemented in
`src/chen/benchmarks/kpis.py` and `src/chen/core/config.py`.

## 1. Cost Model

### 1.1 Per-invocation cost

For a single expert invocation with `i` input tokens, `o` output
tokens, and `P` million parameters loaded:

$$
\text{cost}(i, o, P) = \frac{i \cdot c_\text{in}}{10^6} + \frac{o \cdot c_\text{out}}{10^6} + \mathbb{1}_\text{tax} \cdot \frac{P \cdot (i + o)}{10^3} \cdot c_\text{param}
$$

where:

- $c_\text{in}$ = USD per 1M input tokens (default: 0.15)
- $c_\text{out}$ = USD per 1M output tokens (default: 0.60)
- $c_\text{param}$ = USD per (1M params × 1K tokens) of compute (default: 0.00001)
- $\mathbb{1}_\text{tax}$ = 1 if `include_param_tax=True` (monolith baseline), else 0

The `param_tax` term approximates the cost of loading all parameters
into VRAM for the query — the "monolith tax" CHEN is designed to avoid.

### 1.2 Pipeline-level cost

For a pipeline invoking experts $e_1, e_2, \ldots, e_n$ with parameter
counts $P_1, \ldots, P_n$:

$$
\text{cost}_\text{pipeline} = \sum_{k=1}^{n} \text{cost}(i_k, o_k, P_k)
$$

Note that $\text{cost}_\text{pipeline}$ sums *invoked* parameters, not
*distinct* parameters — if expert $e_1$ is invoked twice, its cost is
counted twice.

### 1.3 Cost per 1M tokens

$$
\text{cost per 1M} = \frac{\text{cost}_\text{pipeline}}{\sum_{k=1}^{n} (i_k + o_k)} \times 10^6
$$

## 2. Effective Parameter Utilization (EPU)

### 2.1 Definition

$$
\text{EPU} = \frac{P_\text{baseline}}{P_\text{distinct}}
$$

where:

- $P_\text{baseline}$ = parameter count of the baseline monolith (e.g. 70,000 for 70B)
- $P_\text{distinct} = \sum_{e \in \text{distinct experts invoked}} P_e$

### 2.2 Interpretation

- **EPU > 1**: CHEN used fewer parameters than the baseline and (we hope) matched its quality.
- **EPU ≥ 3**: the target. If a 14B CHEN swarm matches a 42B monolith, EPU = 3.0.
- **EPU < 1**: CHEN used *more* parameters than the baseline. Routing is wasteful.

### 2.3 Caveats

EPU measures **capacity efficiency**, not quality. A high EPU with low
accuracy means we were efficient but wrong. Always pair EPU with an
accuracy comparison against the baseline.

### 2.4 Capacity-utilization EPU (pipeline-level)

The pipeline also computes a per-run "capacity utilization" EPU:

$$
\text{EPU}_\text{cap} = \frac{P_\text{distinct}}{P_\text{invoked}}
$$

where $P_\text{invoked} = \sum_{k=1}^{n} P_k$ (counts re-invocations).
A value of 1.0 means every distinct expert was invoked exactly once;
<1.0 means experts were re-invoked (routing inefficiency).

## 3. Latency-to-Accuracy Ratio

$$
\text{LTA} = \frac{\text{accuracy}}{L} \times 1000
$$

where $L$ is per-query latency in milliseconds. Higher is better.

CHEN should beat the monolith on LTA for simple queries (where the
router activates few experts) and be competitive on hard queries.

## 4. Latent Nuance Score (Phase 2)

$$
\text{nuance} = \frac{|\{k : \text{transfer}_k \text{ succeeded}\}|}{|\{k : \text{transfer}_k \text{ attempted}\}|}
$$

A value of 1.0 means every KV-cache transfer succeeded; 0.0 means all
failed (the pipeline ran in text-fallback mode).

### 4.1 Limitations

This metric measures *transfer success*, not *nuance preservation*. A
real Phase 2 experiment should also compute the KL divergence between
expert $k$'s output distribution under KV-pass vs. text-pass:

$$
D_\text{KL}\left( p_\text{KV}(\cdot | x) \;\|\; p_\text{text}(\cdot | x) \right)
$$

A small KL means the KV-pass produced similar logits to the text-pass
(suggesting nuance was preserved). A large KL means the KV-pass
produced meaningfully different output (which could be better or
worse — needs the accuracy check).

This KL-based probe is on the roadmap; v0.1 ships only the
success-rate metric.

## 5. Routing Decision (LogisticRouter)

For each expert role $r$, the logistic router computes:

$$
\text{score}_r = \sigma\left( b_r + \sum_{f \in F} w_{r,f} \cdot \phi_f(x) \right)
$$

where:

- $\sigma$ = sigmoid function
- $b_r$ = bias for role $r$
- $w_{r,f}$ = weight for feature $f$ on role $r$
- $\phi_f(x)$ = feature $f$ of prompt $x$ (keyword density, length bucket, code/math indicator)
- $F$ = feature set

The router then:

1. Filters roles with $\text{score}_r \geq \tau$ (threshold, default 0.15).
2. Sorts by score descending.
3. Caps at `max_activation` experts.
4. Forces `force_last_role` (default: SYNTHESIZER) to be the final expert.

## 6. References

- [Mixture of Experts survey](https://arxiv.org/abs/2209.03067) — Fedus et al.
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mistral AI
- [Switch Transformer](https://arxiv.org/abs/2101.03961) — Fedus et al.
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Kaplan et al.
