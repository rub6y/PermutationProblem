# Symmetry search — summary

Starting point: `good_permutation.txt` (n=26, J = 482/975 ≈ 0.4936) has every
column sum `Σ_k σ_k(i) = 75 = 3·25`, i.e. its embedded support points all lie
on the central hyperplane `{x ∈ [0,1]^6 : Σx_i = 3}`. This looked like it
might be a necessary feature of near-optimal measures, by analogy with the
pair-shattering case (k=2), whose unique, proven maximizer is the uniform
measure on the anti-diagonal `{x_1+x_2=1}` — also the central hyperplane of
`[0,1]^2` (see `measure.tex` §2, Obs. 2.9 / Prop. 2.16).

## Claimed symmetry (first draft — wrong)

Initial claim: J is invariant under the full hyperoctahedral group B_6
(order 46080) acting on `[0,1]^6` by permuting the 6 coordinate axes and
independently reflecting *any subset* of them through the center.
Reasoning: the shattering predicate depends only on whether the six
per-coordinate order-codes (elements of S_3) are pairwise distinct: axis
permutation relabels which coordinate holds which code (still injective),
and reflecting one axis composes that axis's code with the order-reversing
permutation ρ of S_3.

**This is false.** Reflecting only *some* axes applies ρ to some coordinates'
codes and the identity to the rest — a pointwise, per-slot recoding of an
injective map `[6] → S_3`, which is not the same as post-composing by one
fixed bijection and need not stay injective. Verified numerically:
transforming `good_permutation.txt` by a partial-axis reflection changes the
exact shattered-triple count (`fw.shattering.count_shattered_triples`) away
from the baseline 8676/17576.

## Corrected symmetry (verified)

The true symmetry group is `G = S_6 × Z_2`, order `6! · 2 = 1440`:

- any permutation of the 6 coordinate axes (precomposition — stays
  injective), **and**
- reflecting **all six** axes simultaneously, `x ↦ (n-1) - x` elementwise
  (post-composition with the single fixed ρ, applied uniformly — stays
  injective).

Reflecting a proper subset of axes is *not* a symmetry.

Verified numerically: `count_shattered_triples` on `good_permutation.txt` is
unchanged (8676/17576) under 5 random pure axis permutations, under the
all-axes flip, and under 5 random permutation+all-flip combinations; it
changes under partial-axis reflection.

Implementation: `fw/symmetry.py` (`permutation_transforms`,
`reflection_transforms` — now just `{id, flip-all}` — `orbit_average`,
`global_reflection_orbit_average`, `permutation_orbit_average`,
`full_orbit_average`).

## Symmetrization experiment: orbit-averaging hurts, badly

Built the orbit-averaged measure `μ_sym = (1/|G'|) Σ_{g∈G'} g_*μ` for
`μ = ι_26(good_permutation.txt)` under three subgroups, and measured J
(exact for the small orbits, Monte Carlo with 2×10^7 samples for the large
ones — MC sanity-checked against the exact baseline: 0.4935 vs 0.4936):

| seed | support | J |
|---|---|---|
| `good_permutation.txt` (baseline) | 26 | **0.4936** |
| global-reflect orbit (`{id, flip-all}`, 2 elts) | 52 | 0.3659 |
| S₆ axis-permutation orbit (720 elts) | 10260 | 0.0572 |
| full `G` orbit (1440 elts) | 16560 | 0.0568 |

Running the existing Frank–Wolfe search (`fw/frank_wolfe.py`, 200 iterations,
default candidate-pool params) from the global-reflect orbit seed climbs
from 0.3659 to 0.4902 in 8 steps, but plateaus **below** the baseline and
does not recover it. Frank–Wolfe from the baseline itself takes 0 steps (no
improving direction found in the restricted candidate pool at these
settings) — consistent with the earlier report that FW starting directly
from `good_permutation.txt` failed to improve on 482/975.

## Why symmetrization hurts

J is a **cubic, non-concave** functional of μ (`𝒥(μ,μ,μ)`, trilinear form).
Averaging μ with rotated/reflected copies of itself does not average the
*value* of J: most of the new cross terms in the expanded trilinear form
pair a point from one orbit copy with a point from a differently
rotated/reflected copy, and such mixed triples mostly fail to shatter. The
correlated structure that makes the 26-point witness score 0.4936 is exactly
what symmetrizing destroys. The "average of good points is at least as
good" intuition needs concavity, which this functional doesn't have.

## Where this leaves the plan

- The `G = S_6 × Z_2` invariance is real, proven, and now numerically
  regression-tested — useful as a correctness check on `fw.objective` /
  `fw.shattering`, and as a genuine (if weaker than first thought) necessary
  condition on any *unique* maximizer of J, should one ever be proven
  unique.
- It is **not** a useful search heuristic: symmetrizing a good witness
  produces a much worse one, and re-optimizing from there does not recover
  the original value with the current Frank–Wolfe setup.
- The "central hyperplane is necessary for near-optimal measures" hunch
  remains unproven; this experiment gives indirect evidence against
  symmetry-averaging as a route to either proving it or exploiting it
  computationally.
- Both the baseline and the symmetrized-then-reoptimized search plateau
  around J ≈ 0.49–0.494, suggesting the current bottleneck is the
  candidate-pool / LP-oracle exploration radius in `fw/candidates.py`
  (`build_candidate_pool`), not the symmetry class of the starting measure.
  A more promising next step is probably widening/restructuring that search
  (larger jump radius, block-swap moves, or accepting temporarily-worsening
  steps) rather than further symmetry-based initializations.
