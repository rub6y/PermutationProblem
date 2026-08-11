# Experiments log (consolidated)

Merged from 9 separate session notes on 2026-08-11 to reduce file count;
content is concatenated in roughly chronological order (each file's own
cross-references to `records_best_structure.md` etc. now mean "the section
below/above in this file"). See `/home/ruba/Documents/Math/KNOWLEDGE.md` for
the compact summary and current best result.

Order: symmetry search -> FW push -> Track 1 diagnosis -> records structure
(n=26) -> recursive construction scaling -> next-steps plan -> warmstart
results (new record) -> records structure (M=78) -> particle swarm.

---


---

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

---

# FW push — summary

Follow-up to `symmetry_search.md`. That investigation found that Frank-Wolfe
(`fw/frank_wolfe.py`) takes 0 steps starting from `good_permutation.txt`
(n=26, J=482/975≈0.493628): the restricted-candidate-pool LP oracle finds no
improving direction. Two hypotheses were raised for why: (1) the candidate
pool (`fw/candidates.py`) was too sparse to see real improving directions,
or (2) the configuration is a genuine local optimum and a wider pool won't
help. This note covers testing both, plus a follow-up block-swap approach.

## 1. Widening the FW candidate pool

`fw/candidates.py`'s `neighbor_points` previously sampled only
`n_neighbors_per_point` random single-coordinate shifts within a small
`radius` around each support point — a small, randomly-subsampled slice of
the true one-axis neighborhood. Replaced it with `full_neighbor_points`:
every single-coordinate move of every support point to every possible value
in `[0, n)`, exhaustively (cheap: gradient evaluation is O(support_size^2)
per candidate, so ~4300 candidates at n=26 costs nothing). Also removed the
now-unused `n_neighbors_per_point`/`radius` params from `fw.frank_wolfe.run`
and `scripts/run_fw_search.py`.

**Result: still 0 improving steps**, but now for a stronger reason. Checked
directly that the LP oracle's returned vertex is *exactly* the original
26-point support, even though several individual candidate points in the
pool have higher raw gradient (1.518) than the support average (1.481) —
the marginal constraints mean no single-point relocation can be traded in
without a matching loss elsewhere. This ruled out hypothesis (1): it's not
a pool-sparsity artifact, `good_permutation.txt` is a genuine local optimum
against every single-point FW move.

## 2. Block-swap local search (`fw/swap_search.py`, new module)

Since single-vertex FW steps can't move more than one point at a time, and
`good_permutation.txt`'s support is literally 6 permutations of `[n]`
(`support_size == n`), a natural move that stays inside `A_n` for free (no
LP needed) is: pick one coordinate and transpose the values at two support
positions — composing `sigma_k` with a transposition. Built:

- `_best_swap` / `_swap_delta`: exhaustively evaluates every `(coord, i, j)`
  transposition and returns the best (numba, parallel).
- `hill_climb`: repeatedly applies the best-improving transposition until
  none improves.
- `basin_hop`: hill-climb to a local optimum, then repeatedly perturb with
  random swaps and hill-climb again, always perturbing from the *best*
  point found so far (not the most recent), keeping the global best.

**Bug caught along the way:** the first version of `basin_hop` perturbed
from the most recently reached local optimum rather than the best one, so
a run of unlucky restarts permanently dragged the walk downhill (observed:
J decayed monotonically from 0.49 to 0.29 as perturbation strength grew,
with no pull back toward the good basin). Fixed to always perturb from
`best_points`.

**Correctness-checked** `_swap_delta` against brute-force recomputation on
15 random swaps (exact match) before trusting any results from it.

## 3. Speeding up `_best_swap` (this session's "option a")

`_best_swap` originally recomputed the full O(n^3) shattered-triple count
before and after each candidate swap. Since a swap only touches rows `i`
and `j`, only the O(n^2) triples containing one of those two rows can
change status. Rewrote `_swap_delta` to rescan only that reduced set
(~3752 vs ~17576 triples at n=26). **Verified correct** against the old
brute-force result on the same 15 random swaps, and against the original
`_best_swap`'s answer on `good_permutation.txt` (same best move, delta=-6
both times). **>100x speedup**: 2.5s → 0.02s per exhaustive scan of all
~1950 candidate transpositions at n=26.

## 4. Larger n via a J-preserving blow-up (this session's "option b")

Constructed an n=52 seed by doubling every point of `good_permutation.txt`
into a twin pair: point `i` -> new points `2i`, `2i+1`, with coordinate `k`
value `2*sigma_k(i)`, `2*sigma_k(i)+1` (same order for every coordinate).
This is provably J-preserving at the seed:
- triples of 3 distinct original points reproduce the exact same order
  structure as before (same shattered/not-shattered outcome), so contribute
  the same ratio;
- triples containing *both* members of one twin pair can never be
  shattered — a twin pair's relative order is identical in every coordinate
  by construction, so at most 3 of the 6 required order-codes are reachable.

Verified numerically: seed J = 0.493628 exactly, matching baseline. A swap
between `2i` and `2i+1` in one coordinate is exactly "un-fixing" that twin
pair's order in that coordinate, so the swap-search neighborhood naturally
explores whether breaking that constant-order seed helps — no separate
construction needed.

## 5. Search results

Basin-hopping with the sped-up oracle, always restarting perturbation from
the best point found:

| n | perturbation strengths tried | trials | outcome |
|---|---|---|---|
| 26 | 3, 6, 10 swaps (fully exhausted, 50 trials each); 15 swaps (partial, ~20 trials) | 170+ | every trial returned to exactly 0.493628 or landed strictly worse |
| 52 (blow-up) | 3 swaps (fully exhausted, 20 trials); 6 swaps (partial, ~12 trials) | 30+ | same pattern |

No trial, at either resolution, ever exceeded baseline.

## Where this leaves the plan

- `good_permutation.txt` is now confirmed a local optimum under three
  independent notions of locality: single-point FW/LP relocation
  (measure.tex Sec. 5 LP oracle), single 2-point/1-coordinate transposition,
  and small-to-moderate bursts of transpositions followed by greedy
  re-optimization, at two different resolutions (n=26 and its J-preserving
  n=52 blow-up).
- This is a real negative result, not a search-implementation deficiency:
  the LP-oracle sparsity hypothesis and the "maybe it just needs bigger
  swaps" hypothesis were both tested directly and ruled out within the
  perturbation strengths explored.
- Local combinatorial search from this particular seed appears to have
  plateaued. Two directions not yet tried: much larger perturbations
  (30+ swaps was reached only briefly on n=26 before this session's time
  budget ran out) with a properly optimized/parallelized `_swap_delta` for
  further headroom; or abandoning perturbation-of-`good_permutation.txt`
  entirely in favor of a structurally different starting witness or a
  theoretical (non-search) push on the reformulation itself.

---

# Track 1 scaling regression — diagnosis

Follow-up to `FWPush.md`/`symmetry_search.md`. `scripts/run_track1_scaling.py`
pushes hyperplane-constrained SA (`fw.hyperplane_annealing.multi_start_anneal`)
to n=30..60, seeded on the central hyperplane via `scripts.gap_survey.
hyperplane_seed`. `results/track1_scaling.json` shows the gap to the
conjectured limit **getting worse**, not better, as n grows:

| n  | J        | gap = 0.5 - J | wall time |
|----|----------|---------------|-----------|
| 26 (`good_permutation.txt`) | 0.493628 | 0.006372 | — |
| 30 | 0.392000 | 0.108000 | 30.1s |
| 34 | 0.401333 | 0.098667 | 34.7s |
| 40 | 0.336563 | 0.163438 | 69.6s |
| 50 | 0.337344 | 0.162656 | 164.2s |
| 60 | 0.322667 | 0.177333 | 372.4s |

This is the opposite of what the $\mathfrak{S}(n)\to 1/2$ conjecture predicts,
and worth understanding before trusting the hyperplane conjecture as a
scaling strategy (measure.tex's new codim-1 Conjecture, added in this
session's step 1).

## Root causes identified (by reading + reproducing)

1. **Fixed step budget, growing search cost.** `run_track1_scaling.py` hard-codes
   `N_STEPS=1000`, `N_RUNS=3` for every n. Wall-clock time nonetheless grows
   ~12x from n=30 to n=60 (30s -> 372s, i.e. roughly $O(n^{3.6})$ empirically)
   because `fw.hyperplane_search._candidate_moves`'s pool size and the cost of
   re-scanning affected triples both grow with n. So the same fixed step count
   covers a shrinking fraction of a search space that's growing much faster --
   larger n gets *less* effective search per step of "difficulty", not more.

2. **The k-cycle generalization is built but never used here.** `fw/hyperplane_annealing.py`
   already implements `anneal_hyperplane_multimove`/`multi_start_anneal_multimove`
   (double-swaps + k-cycles, `k_sizes` configurable) -- exactly the move class
   `FWPush.md` flagged as "not fully explored". `run_track1_scaling.py` imports
   and calls only the plain double-swap `multi_start_anneal`, never the multimove
   variant, so this generalization has had zero effect on the n=30..60 results above.

3. **Naively turning on k-cycles does not fix it under the same fixed budget --
   confirmed empirically.** At n=30, `N_STEPS=400`, `N_RUNS=2` (smaller than
   Track 1's own budget, to keep this check cheap), comparing the two move
   sets from the identical seed (`scripts/diagnose_track1.py`):

   | move set | J | wall time |
   |---|---|---|
   | plain double-swap | 0.364222 | 19.0s |
   | double-swap + 3-cycles (`max_per_group=40`) | 0.345333 | 30.8s |

   Adding k-cycles made this run both **slower per step** (more candidates to
   evaluate) and **worse** (fewer effective steps fit in the same wall time,
   and 2 runs is high-variance). This rules out "just turn on the existing
   k-cycle code" as a free fix -- a larger move set needs a correspondingly
   larger step budget to pay off, which compounds problem (1) rather than
   solving it.

4. **Seeding is the hard way.** `hyperplane_seed` builds a *uniformly random*
   point cloud constrained to the hyperplane (via three independent random
   pairings), i.e. SA starts from something close to $J\approx 0$ every time.
   `FWPush.md` §4 already built and numerically verified a **J-preserving**
   n=52 blow-up of `good_permutation.txt` (twin-doubling every point, proven
   to reproduce J=0.493628 exactly at the seed). Track 1 never uses that or
   any other warm start derived from a known-good smaller witness -- every
   run starts from scratch.

## Where this leaves the plan

None of these are fixed yet; this is diagnosis only, per the "step 3" plan
item ("diagnose why" before trusting hyperplane-constrained search as a
scaling strategy). Concrete next steps, in order of expected payoff per
unit of compute:

- Warm-start n>26 runs from J-preserving blow-ups of `good_permutation.txt`
  (or of the current best `records.jsonl` witness, J=0.49431) instead of
  `hyperplane_seed`'s random construction -- this alone should fix most of
  the n=30 vs n=26 gap, since it starts near a good basin instead of at J≈0.
- Scale `N_STEPS`/`N_RUNS` with n (e.g. proportional to candidate-pool size,
  which is the real cost driver per item 1) rather than holding them fixed.
- If k-cycles are reintroduced, budget them explicitly (e.g. two-phase: plain
  double-swap SA to a local plateau, then a k-cycle pass) rather than pooling
  all move kinds into every single step, since item 3 shows pooling with a
  matched budget is a net loss.
- `scripts/diagnose_track1.py` (new, this session) is a minimal harness for
  this comparison; keep it around and rerun it with matched *effective*
  budgets (time, not step count) once the above changes land, since move
  sets of different sizes aren't comparable step-for-step.

---

# What makes the best `records.jsonl` witness better than the algebraic family?

Follow-up to `track1_scaling_diagnosis.md`, plan step 4. `records.jsonl`'s
best entry (`opt: constraint-preserving`, n=26) scores J=0.494310, slightly
beating `good_permutation.txt` (J=0.493628) and far beating
`algebraic_construction_search.py`'s linear/coprime family (J≈0.2485 at
n=26 -- `results/algebraic_construction_search.json`). Goal: find what
structural feature explains the gap, and whether it's a reusable closed
form.

## Basic checks

- All 6 config columns are genuine permutations of $\{0,\ldots,25\}$, and
  every row sums to exactly 75 = 3·25 -- confirms it sits exactly on the
  central hyperplane $H$ (Conjecture in `measure.tex` §4), same as
  `good_permutation.txt` and every other top record checked in step 1/2.
- **Not a relabeling of `good_permutation.txt`**: after sorting both point
  clouds by coordinate 0 (to align rows canonically), 83 of 156 matrix
  entries still differ. This is a distinct local optimum, not a small
  perturbation of the known one.
- **No pairwise algebraic relation between coordinates**: checked all 15
  coordinate pairs for equality or exact reversal ($\sigma_b = 25-\sigma_a$)
  -- none found. This immediately rules out the kind of structure the
  algebraic family assumes (each of the 3 pairs being a single affine map
  plus its reversal): the winning witness does not pair coordinates that way
  at all.

## Cycle structure of coordinate 0 (as a permutation of $[26]$)

$$
(1\,2)(3\,4\,5)(6\,8)(9\,11\,15\,12)(10\,13)(17\,19\,18)(23\,25\,24)
$$
with 12 fixed points (0, 7, 14, 16, 20, 21, 22, and others). No long cycles,
no arithmetic-progression pattern (contrast with the algebraic family's
$\sigma(i) = (ai+b) \bmod n$, which by construction is close to a single
26-cycle for most coprime $a$). The visible structure is a scatter of small
(2- and 3-element, one 4-element) cycles with no obvious global rule
generating them.

## Conclusion

No closed form was found, and the evidence suggests there probably isn't
one to find at this resolution: the winning configuration looks like the
output of local search that happened to land in a slightly better basin
than `good_permutation.txt`'s, not like a formula. This is consistent with
`opt: constraint-preserving` being a search strategy name, not a
constructive one (unlike `opt: q4-generator`, seen elsewhere in
`records.jsonl`, which *is* a named closed-form family but tops out lower,
same pattern as the algebraic family here).

This changes the practical takeaway for "beating the record": the gain from
0.4936 to 0.4943 came from **more/better local search**, not from
discovering algebraic structure to exploit. It reinforces `track1_scaling_diagnosis.md`'s
recommendation (step 3) rather than adding a competing one: the fastest
path to improving on both fronts is the same fix -- warm-start local search
(hyperplane-constrained hill-climbing/SA) from good known witnesses with a
properly scaled step budget, rather than searching for a closed-form family.
The algebraic-construction line of attack (`algebraic_construction_search.py`)
should be considered a dead end at this resolution unless a fundamentally
different family (not per-pair affine maps) is proposed.

---

# `rec_AI.py`'s recursive construction: a scaling method that actually works

Plan step 5 asked whether the record generator can be pushed past n=26, and
whether it degrades like Track 1 (`track1_scaling_diagnosis.md`) or holds up.
`records.jsonl` itself turned out to be n=26-only, but the repo already
contains an *unrelated, undocumented* generator that answers this question
directly: `rec_AI.py` / `recurence.py`'s `recurence_construction`.

## What it does

Starting from `S1` (`good_permutation.txt`'s 26 points as a tensor `S1_T`)
and `Sym` (all 6 permutations of $(0,1,2)$, reshaped via `from_perms_to_tensor`
into 3 six-vectors `Sym_T[0..2]`), each recursion step replaces every point
$x$ with 3 new points $3x + \text{Sym\_T}[i]$, $i=0,1,2$ -- a base-3 digit
expansion using the `Sym_T` vectors as digits. One step multiplies the point
count ($M$) by 3 and the resolution by 3.

## It preserves the central hyperplane exactly -- provably, not just empirically

Computed the three digit vectors: `Sym_T[0]=(0,0,1,1,2,2)`, `Sym_T[1]=(1,2,0,2,0,1)`,
`Sym_T[2]=(2,1,2,0,1,0)` -- **each sums to exactly 6**. If $x$ already satisfies
$\sum_k x_k = 3(M-1)$ (on hyperplane $H$, cf. `measure.tex` Conjecture 4.1),
then $\sum_k (3x_k + \text{Sym\_T}[i]_k) = 3\cdot 3(M-1) + 6 = 3(3M-1)$, which
is exactly the hyperplane target for the new size $3M$. So **every point at
every recursion depth sits exactly on $H$, by construction** -- not
approximately, and not something that needs checking numerically. This is a
genuine closed-form, hyperplane-preserving family, unlike
`algebraic_construction_search.py`'s linear family (which sits on $H$ by
construction too, but scores far lower).

## It scales -- and J goes up, not down

Ran `recurence_construction(S1_T, depth)` + `compute_J_fast` (both already
in `rec_AI.py`) for depth 0/1/2:

| depth | M   | J        |
|-------|-----|----------|
| 0     | 26  | 0.493628 |
| 1     | 78  | 0.493956 |
| 2     | 234 | 0.493993 |

J increases monotonically with n here, converging upward toward (but still
below) 0.5 -- the correct direction per the $\mathfrak{S}(n) \to 1/2$
conjecture, and the **opposite** of Track 1's SA-based scaling attempt,
which got worse from n=26 to n=60 (`track1_scaling_diagnosis.md`). A leftover
run already in `rec_AI.py` (`recurence_construction(S1_T, 5)`, M=6318) gives
J=0.493997, consistent with continued slow convergence.

## Where this leaves the plan

This is the single most useful result of the whole investigation for
"pushing past n=26": a provably hyperplane-preserving, genuinely scaling
construction already exists in the repo, was apparently run once (the
leftover depth-5 call and stray print at the bottom of `rec_AI.py`), and
was never written up or connected to the hyperplane conjecture or to
Track 1's regression. Concrete next steps:

- Investigate *why* it works: the per-step gain (0.493628 -> 0.493956,
  +0.000328) is shrinking each step -- worth checking whether it plateaus
  near 0.494 (like the plain SA record) or keeps climbing slowly toward 0.5;
  a few more depths (3, 4 -- M=702, 2106) would show the trend, at the cost
  of $O(M^3)$ compute (already the bottleneck: depth 5's M=6318 run alone
  took several minutes of parallel numba compute).
- `rec_AI.py` has a name suggesting an LLM-assisted construction ("AI") and
  currently runs its expensive depth-5 example as *module-level* code with
  no `if __name__ == "__main__":` guard -- anything that imports the module
  (as this investigation's own quick check did) re-triggers it silently.
  Should be guarded before the module is reused elsewhere.
- Given this scales cleanly, it's a better base for the warm-start
  suggested in `track1_scaling_diagnosis.md` than either
  `hyperplane_seed`'s random construction or `FWPush.md`'s twin-doubling
  blow-up: run hyperplane-constrained local search *on top of* a
  `recurence_construction` seed at each target n, instead of from scratch.

---

# Plan: push `rec_AI.py`'s recursive construction toward the conjectured limit

**Superseded as entry point by `recurrence_warmstart_results.md`** (this
session executed steps 1-5 below; read that file first -- it has a new best
record, J=0.495195 at M=78, beating the previous 0.494310, found by
warm-starting `fw/hyperplane_search.py`'s `basin_hop_hyperplane` from a
depth-1 `recurence_construction` seed). The rest of this file is kept for
its background/context, oldest to newest: `symmetry_search.md`, `FWPush.md`,
`track1_scaling_diagnosis.md`, `records_best_structure.md`,
`rec_construction_scaling.md`, then `recurrence_warmstart_results.md`, then
`records_structure_m78.md` (structural analysis of the new M=78 record:
mixes two structurally distinct near-optimal families -- long-cycle,
inherited from `good_permutation.txt`, vs. the older record's scattered
small cycles).

## Context (why this, why now)

`measure.tex` conjectures $\mathfrak{S}(n) \to 1/2$, reformulated as
$\sup_\mu \mathcal{J}(\mu)$ over measures on $[0,1]^6$ (§1), and a new
Conjecture (§4, added this project) says any maximizer concentrates on the
central hyperplane $H=\{\sum x_k=3\}$ -- well-supported empirically but
unproven (attempted proof blocked, see `measure.tex` §4 discussion of
multi-marginal OT duality).

Two scaling strategies were tried to push witnesses past n=26
(`good_permutation.txt`, J=0.493628):
- **Track 1** (SA on hyperplane-preserving swaps, `scripts/run_track1_scaling.py`):
  **fails** -- J gets *worse* as n grows (0.39 at n=30 down to 0.32 at n=60).
  Root causes diagnosed in `track1_scaling_diagnosis.md`: fixed step budget
  vs. growing search cost, an unused-but-built k-cycle move set that isn't a
  free win either, and cold random seeding instead of warm-starting.
- **`rec_AI.py`'s `recurence_construction`** (previously undocumented):
  **works** -- a base-3 recursive digit-expansion of `good_permutation.txt`
  that is *provably* exactly on $H$ at every depth (each of the 3 digit
  vectors `Sym_T[0..2]` sums to exactly 6, so the hyperplane identity is
  preserved algebraically under $x \mapsto 3x + d_i$), and empirically J
  *increases* with depth: 0.493628 (n=26) -> 0.493956 (n=78) -> 0.493993
  (n=234) -> 0.493997 (n=6318, leftover run in the file). This is the
  opposite trend from Track 1 and the most promising lead found so far.

This session's goal: understand and exploit `recurence_construction` to try
to push J meaningfully higher / closer to $1/2$, and/or extract the
algebraic reason it works (which could feed back into `measure.tex` as an
actual proof lead, not just a numeric conjecture).

## Concrete next steps, roughly in priority order

1. **Guard `rec_AI.py` before touching it further.** It currently runs an
   expensive (M=6318, several-minutes, unbounded CPU) computation as bare
   module-level code with no `if __name__ == "__main__":` guard -- it fires
   on `import`. Fix this first or every subsequent experiment will
   accidentally re-trigger it.

2. **Map the convergence curve properly.** Only depths 0/1/2 (M=26/78/234)
   were measured, plus a stray depth-5 run (M=6318, J=0.493997). Get depths
   3, 4 too (M=702, 2106) to see the shape of convergence: is it converging
   to something below 0.5 (a plateau, meaning the construction alone can't
   reach the conjectured limit), or slowly climbing toward 0.5? The per-step
   gain is shrinking (0.000328, then 0.000037 from depth 1->2) -- fit that
   decay and extrapolate before spending compute on deeper recursions
   (cost is $O(M^3)$, so depth 5's M=6318 alone took several minutes; depth
   6 would be M≈19000, likely far too slow for exhaustive
   `compute_J_fast` -- may need `fw.shattering`'s existing sparser/reduced
   evaluators, or a sampling-based J estimator, beyond this depth).

3. **Understand *why* it works -- look for an algebraic proof, not just a
   number.** The digit vectors `Sym_T[0..2]` (each summing to 6, built from
   3 of the 6 elements of $S_3$) are the mechanism. Worth checking: does
   using all 6 `Sym` rows instead of 3 (i.e. `N=6` in
   `recurence_construction`, multiplying by 6 instead of 3 per step) still
   preserve $H$ and score at least as well? If the construction's gain comes
   from a genuine self-similarity/measure-limit argument, it may be provable
   directly, which would be a real contribution to `measure.tex` §4's open
   conjecture (the measure-theoretic limit of the recursive point clouds,
   as depth -> infinity, would be a concrete candidate maximizer measure --
   worth deriving its closed form as a self-similar/IFS-type measure on $H$
   and computing $\mathcal{J}$ of it directly, rather than only via finite-n
   `compute_J_fast`).

4. **Use it as the warm start Track 1 needs.** `track1_scaling_diagnosis.md`
   already recommends warm-starting hyperplane-constrained local search from
   a good witness instead of `hyperplane_seed`'s random construction.
   `recurence_construction`'s output at a given depth is a strictly better
   candidate seed than the twin-doubling blow-up in `FWPush.md` (that one
   was J-preserving but flat; this one already gains). Try:
   hyperplane-constrained hill-climb/SA (`fw/hyperplane_search.py`,
   `fw/hyperplane_annealing.py`) starting from `recurence_construction`
   depth-2 or depth-3 output, to see if local search can push past 0.494
   faster than either pure recursion or pure SA-from-scratch alone.

5. **If a genuinely better witness is found at any n, record it** in
   `records.jsonl`-compatible form and note the `opt` provenance (e.g.
   `"opt": "recurrence+hillclimb"`) so future sessions can mine it the way
   step 4/5 mined the existing best record.

## Reusable code already in place (don't re-derive)

- `PermutationProblem/rec_AI.py` -- `S1_T`, `Sym_T`, `recurence_construction`,
  `compute_J_fast` (numba parallel, exact, $O(M^3)$).
- `PermutationProblem/fw/hyperplane_search.py`,
  `fw/hyperplane_annealing.py` -- hyperplane-preserving local search
  (hill-climb, SA, k-cycle moves), ready to seed with any point cloud.
- `PermutationProblem/fw/shattering.py` -- `count_shattered_triples`,
  `is_shattered_triple`, the shared low-level primitives.
- `nix-shell shell.nix --run "python3 ..."` is the standard way to run
  anything in this repo (numpy/numba/scipy/matplotlib pinned there).

## Caution learned this session

`nix-shell` invocations can hang or silently block-buffer output when piped
(`| tail`) -- prefer `python3 -u` and/or redirecting to a log file plus
polling, and always background anything that might run past ~1 minute
rather than blocking the session on it.

---

# Recurrence construction: convergence curve, and a new record via warm-start

Follow-up to `NEXT_recurrence_scaling_plan.md` (read that first for full
background). Executes plan steps 1-5 (guard `rec_AI.py`, map the convergence
curve, check the "N=6 variant" idea, warm-start local search from a
recursion seed, record any improvement).

## 1. `rec_AI.py` guarded

Added `if __name__ == "__main__":` around the module-level depth-5 run.
Also generalized `recurence_construction(T, n, digits=None)` to take an
arbitrary digit set (default unchanged: `Sym_T[:3]`) -- needed for step 3.

## 2. Convergence curve (depths 0-4)

`scripts/recurrence_convergence.py`, exact `compute_J_fast`:

| depth | M    | J        | gain       | gain ratio |
|-------|------|----------|------------|------------|
| 0     | 26   | 0.493628 | --         | --         |
| 1     | 78   | 0.493956 | +0.000328  | --         |
| 2     | 234  | 0.493993 | +0.000037  | 0.113      |
| 3     | 702  | 0.493997 | +0.0000037 | 0.100      |
| 4     | 2106 | 0.493997 | ~0         | --         |

Every depth confirmed exactly on $H$ (row sums $=3(M-1)$ for all $M$, exact
integer check, not a numeric tolerance). The gain shrinks geometrically with
ratio $\approx 1/9 = 1/3^2$ per step -- plausible given the construction is a
base-3 digit expansion (a 3x linear rescaling per depth), so a second-order
correction term decaying like the square of the contraction ratio is the
natural guess. Extrapolating the geometric tail from depth 3 gives
$J_\infty \approx 0.493998$, i.e. **this specific construction plateaus
strictly below the pre-existing best record (0.494310)**, not above it. This
is a negative result for the recursion *by itself*: it does not approach 0.5,
and it doesn't even reach the best previously known finite-$n$ witness.

## 3. The "N=6 variant" idea from the plan doesn't apply as stated

`NEXT_recurrence_scaling_plan.md` step 3 asked whether using all 6 rows of
`Sym_T` instead of 3 still works. Checked directly: `Sym_T` only *has* 3
rows. `Sym` lists all 6 elements of $S_3$ (each a 3-tuple), and
`from_perms_to_tensor` transposes by *position within each 3-tuple* (3
positions), not by which group element it is -- so `Sym_T` is inherently a
3-row object, not a 6-row one with an arbitrary 3-row subset taken. There is
no size-6 analogue sitting unused in the same tensor.

What *is* true, and is the real structural reason the construction preserves
$H$: `Sym_T[i]` (for $i \in \{0,1,2\}$) is, as a multiset, always
$\{0,0,1,1,2,2\}$ -- each of the 3 values appears in position $i$ of exactly
2 of the 6 elements of $S_3$ (since $|S_3|/3 = 2$). This holds for *any*
ordering of the 6 group elements in the `Sym` list, so the digit-vector-sums-
to-6 property (hence hyperplane preservation) is forced by $S_3$ acting
transitively-and-evenly on each coordinate slot, not by the particular
listed order. This is the reusable structural fact for step 6, not an N=6
generalization.

## 4. Warm-starting local search from a recursion seed: beats the record

`scripts/recurrence_warmstart.py` seeds `fw.hyperplane_search.
basin_hop_hyperplane` (30 restarts, `perturb_swaps=3`, `seed=0`) from
`recurence_construction(S1_T, depth)` instead of Track 1's random
`hyperplane_seed` (diagnosed cause of Track 1's regression:
`track1_scaling_diagnosis.md`).

| depth seed | M   | seed J   | basin_hop best J | vs. old record (0.494310) |
|------------|-----|----------|-------------------|----------------------------|
| 1          | 78  | 0.493956 | **0.495195**      | beats it (+0.000885) |
| 2          | 234 | 0.493993 | not obtained this session; two attempts killed, see "Left for later" below |

The depth-1 result was reproduced deterministically (`seed=0` in both
`basin_hop_hyperplane` and the RNG), independently re-verified with
`fw.shattering.count_shattered_triples` (not just `exact_J`, which calls the
same function -- cross-checked to guard against implementation drift), and
confirmed exactly on $H$. **New best known witness: J=0.495195 at M=78**,
appended to `records.jsonl` (`"opt": "recurrence-depth1+basin_hop"`,
`"seed": 0`) via `scripts/save_recurrence_witness.py`.

This confirms `track1_scaling_diagnosis.md`'s standing recommendation: the
problem with Track 1 was never the hyperplane constraint or n itself, it was
cold-starting from J≈0. A recursion-seeded warm start, even at the cheap
depth-1 (M=78), immediately both scales past n=26 *and* beats the best n=26
record.

## Left for later

- Depth-2 (M=234) warm-start didn't finish in this session, in two attempts:
  first with the original 30-restart/1000-step budget (killed after ~40
  CPU-min), then with a reduced 4-restart/150-step budget
  (`scripts/recurrence_warmstart_depth2.py`) that still didn't print even a
  single hill-climb result after ~20 min wall / 70+ CPU-min. This rules out
  "just lower the step cap" as a fix: cutting `max_hill_climb_steps` from
  1000 to 150 (6.7x) did not produce a proportionate speedup, which means
  the bottleneck isn't the number of hill-climb steps taken but the
  **per-step cost**, and that cost is dominated by
  `fw/hyperplane_search.py`'s `_candidate_moves` -- a pure-Python function
  that builds a Python list of every hyperplane-preserving double-swap
  candidate ($O(M^2)$ pairs $\times$ up to 15 coordinate pairs) from
  scratch on *every* hill-climb step, before handing it to the numba-jitted
  evaluator. At $M=78$ this is small enough to be dominated by the fast
  numba path (30 full restarts finished in 24s); at $M=234$ ($\approx 9\times$
  more candidate pairs) the pure-Python construction itself appears to
  dominate wall time, consistent with `track1_scaling_diagnosis.md`'s
  original diagnosis of `_candidate_moves` as the real cost driver, now
  confirmed at a second, larger $M$.
  **Before attempting depth-2 again**, `_candidate_moves` needs to be
  vectorized (numpy broadcasting or a numba-jitted candidate builder) rather
  than just tuning restart/step-count budgets -- that's the actual fix, not
  a search-schedule change. `fw/hyperplane_annealing.py`'s SA variant likely
  has the same underlying bottleneck if it also calls `_candidate_moves` per
  step; check before assuming it would be faster.
- Given depth-1 already beats the old record, it's worth also trying
  depth-0 (M=26, i.e. `good_permutation.txt` itself) through the same
  `basin_hop_hyperplane` call with a larger restart budget, to check
  whether the gain is really coming from the depth-1 recursion structure or
  just from basin-hopping being a stronger search than whatever produced the
  0.494310 record in the first place -- this matters for correctly
  attributing the improvement before writing anything stronger into
  `measure.tex`.
- No closed-form limiting measure was derived (the plan's more ambitious
  step 6 goal). Given §2/§3's negative result (this recursion's own limit is
  provably not the maximizer, since local search already beats its
  plateau), deriving its IFS limit in closed form is no longer the most
  promising path to a `measure.tex` §4 proof lead -- **not** added to
  `measure.tex`. The one fact worth keeping for a future write-up, if this
  line is picked up again, is §3's structural point (any listing-order of
  $S_3$'s 6 elements gives hyperplane-preserving digit vectors, because
  each coordinate slot sees each of the 3 values exactly twice) -- a clean
  finite-group fact, but on its own it doesn't yet imply anything about
  Conjecture "codim-1 concentration", which remains open.

---

# Structural scan of records.jsonl after the new M=78 record

Follow-up to `records_best_structure.md` (which analyzed the earlier best
M=26 record) and `recurrence_warmstart_results.md` (which found the new
M=78 record, J=0.495195). `records.jsonl` also turned out to contain two
previously-unexamined algebraic-family entries at M=104 and M=936
(`q4-generator`, `q6-generator`) from `algebraic_construction_search.py`.
`scripts/analyze_m78_record.py` does the analysis below; rerun it for the
raw numbers.

## Cross-record summary

| M   | opt                    | J        | on H |
|-----|-------------------------|----------|------|
| 26  | constraint-preserving   | 0.494310 | yes  |
| 78  | recurrence-depth1+basin_hop | **0.495195** | yes |
| 104 | q4-generator            | 0.494182 | yes  |
| 936 | q6-generator            | 0.494219 | yes  |

The two closed-form algebraic families (`q4`/`q6-generator`) plateau around
J≈0.4942 *even at M=936* -- below both the old and new best records despite
having 10-35x more points. This reconfirms `rec_construction_scaling.md`'s
earlier point about `algebraic_construction_search.py`'s family: closed-form
constructions top out lower than search-refined witnesses; raw point count
doesn't buy J on its own.

## The M=78 record is only ~38% "pure recursion"

Compared the M=78 record's 78 points against the untouched depth-1
`recurence_construction(S1_T, 1)` output it was seeded from (same row
order, since `basin_hop_hyperplane` never reorders points, only swaps
coordinate values pairwise):

- **30 of 78 points (38%) are byte-identical to the pure recursion output.**
- **48 points (62%) were relocated** by `basin_hop_hyperplane`'s
  hyperplane-preserving double-swaps -- removed as one 6-tuple, added back
  as a different one, always in matched pairs (both counts are 48, as
  double-swaps move value pairs between exactly two points at a time,
  cascaded over many search steps).

So the +0.0025 gain over the recursion's own plateau (0.493956) needed
reworking a *majority* of the point cloud, not a small local nudge -- this
is a genuinely different local optimum nearby in search space, not a minor
correction to the recursive family.

## Cycle structure: two distinct structural families reach similar J

`records_best_structure.md` found the earlier M=26 "constraint-preserving"
record has **no long cycles** in any coordinate -- a scatter of 2-, 3-, and
one 4-element cycle, 12 fixed points, "no obvious global rule."

The new M=78 record looks structurally *very different*: coordinates 3 and
5 each have one **near-Hamiltonian long cycle** (length 43 and 38 out of 78
points respectively, i.e. covering more than half the coordinate's points),
plus assorted shorter cycles elsewhere, and very few fixed points on
coordinates 3-4 (0 fixed points each).

Tracing this back: **the long cycles are not new** -- `good_permutation.txt`
(`S1`, the un-recursed depth-0 seed) already has a 23-cycle on coordinate 5
(out of 26 points) and a 12-cycle on coordinate 3, alongside near-identity
or small-cycle structure on the other four coordinates. The base-3 digit
recursion roughly triples the ambient point count per step; a coordinate's
cycle structure gets carried through the digit expansion (mapped through
$x \mapsto 3x+d$) and comes out close to 3x the original cycle length
(23 -> 38, 12 -> 43 after `basin_hop`'s mild perturbation on top). Coordinate
0 is the odd one out: since `S1`'s coordinate 0 is literally the identity
permutation $(0,1,\ldots,25)$, the pure recursion output for coordinate 0 is
also (close to) the identity on $[0,77]$; `basin_hop` introduces a modest
scatter of small cycles there (10 cycles, longest length 4) -- structurally
closer to the *other* family (M=26 constraint-preserving) than to its own
sibling coordinates 3/5.

**Conclusion:** there are at least two structurally distinct families of
near-optimal witness -- "long-cycle" (S1/recursion-derived, ~0.4936-0.4952)
and "scattered small-cycle, no long cycle" (constraint-preserving,
~0.4943) -- reaching comparable J by different combinatorial routes, and a
single witness (the M=78 record) can even mix both patterns across its own
six coordinates. This reinforces `records_best_structure.md`'s standing
conclusion: cycle structure/shape is not what determines near-optimality,
and there is no evidence of a single "right" combinatorial form to search
for -- the improvement is coming from local-search basin quality, not from
matching some particular cycle-structure template.

## PCA sanity check (nothing new, confirms exact H-membership)

Covariance eigenvalues of the M=78 point cloud: one eigenvalue is exactly 0
(to floating-point precision) with eigenvector cosine-similarity 1.0 against
the all-ones direction $(1,\ldots,1)/\sqrt6$ -- the expected signature of
sitting exactly on $H$, consistent with the algebraic proof in
`rec_construction_scaling.md` and the direct integer row-sum check. Not a
new finding, included only as a regression check.

---

# Particle-based search: two approaches tried, tested, and their behavior

Code: `scripts/particle_swarm_search.py`. Raw data:
`results/particle_swarm_experiments.json` (n=26 cases, and the original,
methodologically-flawed n=78 single-trial run), `results/particle_swarm_n78_timeboxed.json`
(the corrected, 5-seed n=78 comparison -- see "n=78 update" below). Plots:
`plots/particle_n26_fresh.png`, `plots/particle_n26_warmstart.png`,
`plots/particle_n78_fresh.png` (this last one is from the superseded run;
not regenerated for the timeboxed re-run).

**Update:** the original n=78 section below (single trial, wall-clock gated)
turned out to have two compounding measurement bugs. Both are fixed in
`particle_swarm_search.py` now; see "n=78 update: methodology fixed, re-run
with 5 seeds" for the corrected numbers and what changed. The rest of this
file (n=26 results, method descriptions) is unaffected and still accurate.

## Context

Idea to test: a population of "particles" scattered around the search space,
moving via simple local steps, as an alternative to the existing
`fw.hyperplane_search.basin_hop_hyperplane` (exhaustive best-improving swap +
basin-hopping). The feasible set (6-tuples of permutations of `[n]`,
restricted to the central hyperplane `sum_k sigma_k(i) = 3(n-1)`) is a finite,
disconnected lattice, not a vector space -- there is no continuous gradient
step available. So every "particle move" here is a single discrete
hyperplane-preserving double-swap; the two approaches differ only in *how*
moves are chosen and how many walkers run.

Two approaches were implemented, reusing the project's existing move set and
exact functional-derivative code (nothing about the swap machinery was
reimplemented):

- **`population_swarm`**: 8 independent point-cloud candidates
  ("pseudo-randomly scattered points"), each doing 1 Metropolis swap step per
  round, culled every 40 rounds (worst half replaced by mutated clones of the
  best half) -- a discrete stand-in for PSO's "move toward the best particle,"
  since there's no velocity vector between two permutations to interpolate.
- **`gradient_biased_walk`**: a single point cloud where
  `fw.objective.compute_gradient` (the same exact functional derivative the
  Frank-Wolfe LP oracle already uses) ranks the current points by
  contribution to J every 10 steps, and swap proposals are restricted to the
  lowest-contributing quarter. The gradient only *ranks* discrete moves; it
  never displaces a point continuously.

Baseline: `basin_hop_hyperplane` from the same seed, given the same wall-clock
budget (not the same step count -- its per-step cost is much higher, since it
exhaustively scores every hyperplane-preserving move each round, so equal
step counts would be an unfair comparison).

## Results

| Case | seed J | baseline (basin_hop) | population_swarm | gradient_biased_walk |
|---|---|---|---|---|
| n=26, fresh random seed, 20s budget | 0.213359 | 0.354347 | **0.419549** | 0.383364 |
| n=26, warm-started from `good_permutation.txt`, 20s budget | 0.493628 | 0.493628 | **0.493969** | 0.493628 |
| n=78, fresh random seed, 60s budget | 0.218771 | **0.291370**\* | 0.266677 | 0.329717\*\* |

\* baseline ran 86.9s wall time (it can only be time-boxed between restarts,
not mid-restart) -- 27s over budget, a real handicap against it here.
\*\* not reproducible in a repeat run with the same seed (see Caveat below);
treat this cell as noisy, not a confirmed win.

### n=26, warm-started: the "improvement" is a known result, not new

`population_swarm` found J=0.493969, beating the seed's 0.493628. Verified
independently (`fw.swap_search.exact_J`, plus checked all 6 columns are
still valid permutations of `range(26)`): **this value is bit-identical**
(0.4939690487027765) to `results/annealing_best_n26.txt`, a witness already
on disk from an earlier `run_annealing.py` session. So `population_swarm`
independently re-derived an already-known local optimum rather than finding
new ground -- a useful consistency check on the implementation, not a new
record. (For context, `records.jsonl`'s stored n=26 best is 0.494310, from a
different, non-hyperplane-constrained search family -- not beaten here.)

### n=26, fresh start: gradient/population methods clearly beat exhaustive local search under a time budget

From a cold random seed, both new methods reach a materially better J than
`basin_hop_hyperplane` in the same 20 seconds (0.42 and 0.38 vs 0.35).
Mechanism: `basin_hop_hyperplane`'s exhaustive per-round move evaluation is
O(n^2) moves x O(n^2) delta cost each round; a single cheap random/ranked
swap is O(n^2) once. In a fixed time budget, "many cheap steps" explored more
of the space than "few exhaustive steps" at this n.

### n=78: inconclusive -- variance dominates

The headline number (`gradient_biased_walk` at 0.3297, beating baseline's
0.2914) did **not** reproduce in an immediate same-seed repeat (0.2697
instead). Root cause: the stopping condition is wall-clock time, and the
number of accepted steps that fit in a fixed time budget depends on machine
load and Numba JIT warm-state at call time (first call in a process pays
compilation cost; a later call in the same process doesn't) -- so two "same
seed" runs can execute a different number of steps and diverge. This is a
methodology gap, not a property of the algorithm: single-trial, wall-clock
gated comparisons aren't reliable enough to declare a winner at this n yet.

## n=78 update: methodology fixed, re-run with 5 seeds

Two bugs were found and fixed, in order:

1. **Numba JIT warm-up wasn't controlled for.** Whichever method ran first in
   a fresh process paid Numba's one-time compilation cost out of its own
   timed budget; a second process running the same seed took a different
   number of steps and landed at a different J (0.330 vs 0.270 for
   `gradient_biased_walk`, seed 3, same 60s budget). Fixed by `_warmup_jit()`,
   called once before any timing starts, that exercises every hot
   `@njit` path (`_double_swap_delta`, `count_shattered_triples`, the
   `compute_gradient` kernel, `hill_climb_hyperplane`'s parallel move
   evaluator) on a throwaway n=10 cloud.
2. **The baseline wasn't actually time-boxed.** `basin_hop_hyperplane_timed`
   only checked the clock *between* restarts; `hill_climb_hyperplane`'s
   first call, uninterrupted, took ~100s from a cold n=78 seed by itself --
   4x the nominal 25s budget -- while the two particle methods were correctly
   cut off at 25s. Fixed by `_hill_climb_hyperplane_timed`, which checks the
   budget after every individual swap step, not just between restarts.

With both fixed, 5 seeds each (seeds 100-144, offset per trial and per
method so trials aren't correlated across methods), true 25s budget for all
three methods:

| method | mean J | std | min | max |
|---|---|---|---|---|
| baseline_basin_hop | **0.279346** | 0.000128 | 0.279143 | 0.279548 |
| population_swarm | 0.262944 | 0.011194 | 0.250367 | 0.278878 |
| gradient_biased_walk | 0.253181 | 0.001830 | 0.251403 | 0.256461 |

This reverses the earlier (buggy) headline finding that `gradient_biased_walk`
beat baseline at n=78 (that was baseline running on 25s while it was
secretly given ~100s). With a real equal budget, **baseline
`basin_hop_hyperplane` wins reliably** at n=78, mean J higher than both
particle methods by more than either method's std. `population_swarm` is a
distant second and its highest trial (0.2789) nearly reaches baseline's
range, worth another look with more seeds/tuning; `gradient_biased_walk` is
consistently the weakest of the three at this n and budget, the opposite of
what the earlier, buggy comparison suggested.

Net: the n=26-cold-start finding (many cheap steps beat exhaustive search
under a small time budget) does **not** hold at n=78 with a 25s budget --
exhaustive local search's higher per-step quality wins once each cheap-step
method's steps are numerous enough to matter less than their lower
per-step improvement rate. Where the crossover point is (some n and budget
between 26 and 78) is not yet known.

## Caveats (read before trusting any single number above)

1. ~~Single trial per (method, n, seed) cell~~ **fixed for n=78** -- now 5
   seeds/method via `run_experiment_multiseed`, std reported. The n=26 cases
   are still single-trial; if they matter for a future decision, re-run them
   the same way first.
2. ~~Wall-clock-based stopping couples results to machine/JIT state~~
   **fixed**: `_warmup_jit()` runs before any timing, and
   `_hill_climb_hyperplane_timed` makes the baseline's stopping condition
   actually check the clock mid-climb instead of only between restarts. Both
   bugs are described in detail in "n=78 update" above.
3. All witnesses were verified valid (each of the 6 columns really is a
   permutation of `range(n)`) and J was independently recomputed with
   `fw.swap_search.exact_J`.
4. Only 3 n values tested (26 fresh, 26 warm, 78 fresh); no sweep of
   `population_swarm`'s particle count/cull schedule or
   `gradient_biased_walk`'s `top_k_frac`/`regrad_every` -- these were picked
   once, not tuned. `population_swarm`'s n=78 std (0.011) is an order of
   magnitude larger than the other two methods' -- worth understanding
   before trusting its mean, e.g. is one seed finding a genuinely different
   basin, or is 5 trials just too few.
5. The n=26 cases have the same "un-timed hill-climb" and "no JIT warm-up"
   exposure that n=78 had -- they weren't re-run because the n=26 baseline
   converges fast enough there that neither bug plausibly changes the
   conclusion (confirmed: baseline's n=26 warm-start result matches a
   known-good witness to 16 significant figures), but this wasn't verified
   with a multi-seed re-run the way n=78 was.

## What this suggests for next steps

- The n=26-cold-start finding stands: cheap ranked/random single-swap steps
  beat exhaustive best-swap search there under a small time budget. But this
  **does not generalize to n=78** -- with the timing bugs fixed, exhaustive
  `basin_hop_hyperplane` wins clearly and reproducibly at n=78 (mean 0.2793
  vs 0.2629 and 0.2532, both gaps bigger than either loser's std). There is
  a crossover somewhere between n=26 and n=78 where exhaustive search's
  higher per-step quality starts to outweigh the particle methods' higher
  step-count; finding it (e.g. n=40, 50, 60) is the natural next experiment,
  not scaling straight to production n.
- `population_swarm`'s culling step is doing very little differentiated work
  yet (same move-proposal distribution as a plain multi-start SA) -- if
  pursued further, the natural next test is whether crossover/cloning
  actually helps versus N independent SA chains with no interaction at all
  (an ablation this session did not run). Its high n=78 variance (std=0.011,
  vs 0.0002 and 0.0018 for the other two) makes it the most likely of the
  three to reward tuning (cull schedule, mutation strength) or more seeds.
- `gradient_biased_walk` was the strongest particle method at n=26 but the
  weakest at n=78 -- opposite of what the (buggy) first n=78 run suggested.
  Worth checking whether `top_k_frac=0.25`/`regrad_every=10` (picked once,
  never tuned) is simply a bad fit at larger n before concluding the
  approach doesn't scale.
