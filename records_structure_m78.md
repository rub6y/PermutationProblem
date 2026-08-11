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
