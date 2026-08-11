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
