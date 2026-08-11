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
