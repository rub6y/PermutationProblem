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
