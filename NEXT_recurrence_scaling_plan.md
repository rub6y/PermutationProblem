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
