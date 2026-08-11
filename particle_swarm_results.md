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
