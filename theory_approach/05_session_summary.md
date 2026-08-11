# Session summary: constructions tried toward J -> 0.5

Goal was to find a sequence of measures pushing J toward 0.5, building on
the pre-existing best record (J=0.495195, M=78) and the H-concentration
conjecture (`measure.tex` S4).

## Findings, in order

1. **Fixed-gadget recursion is provably a dead end** (pre-existing finding,
   confirmed): the S3-digit-expansion IFS plateaus at J~=0.493998, below
   even the record. Any fixed small self-similar gadget, iterated, converges
   to its own limit measure's J -- it cannot climb toward 0.5 by definition.

2. **Reflection-pair algebraic family: proven dead end via exact reduction**
   (`03_torus_reflection_pairs.md`). Its continuum limit collapses, via the
   S3 reversal involution, to a 2-parameter equidistribution problem; caps
   near J~=0.22-0.25 for any slope choice, matching the discrete family's
   reported ~0.2485. No hidden regime to search into.

3. **Circular construction (d=2 order types)**: validated (2/9 baseline,
   two independent methods agree), then fully optimized (joint annealing
   over point measure and 6 directions) -- caps at J~=0.2498 across
   n=60/80/100, tight agreement across seeds. Falsified as competitive.

4. **Sphere construction (d=3)**: baseline already matches d=2's optimized
   ceiling with zero search; annealing pushes to J~=0.257 at n=70, still
   rising, not yet plateaued. Real but small improvement -- a two-point
   (d=2,3) trend can't tell "climbing to 0.5" from "climbing to ~0.3".
   Left open; would need a proper d-sweep to resolve.

5. **Independence/pseudorandomness (iid permutations, discrete-log
   family)**: actively harmful, J~=0.008-0.017, near the theoretical
   6!/6^6~=0.0154 floor. Confirms all gain in every viable construction
   comes from forcing correlation near H, not from diversifying the 6
   permutations.

6. **LP-oracle stationarity check on the record** (`02_perturbative_expansion.md`):
   the M=78 record is an exact first-order stationary point of J restricted
   to A_78 against a 56k-point candidate pool (single-coordinate moves +
   random resampling). No slack left at this resolution via local moves.

## Net position

Every alternative generative mechanism tried this session -- recursion,
two distinct algebraic/geometric families, and pure randomness -- tops out
between 0.015 and 0.257, all well below the record (0.495) let alone 0.5.
The record itself has no exploitable first-order local slack. Nothing found
this session moves the lower bound past 0.495195, and nothing here
constitutes progress toward proving J -> 1/2; the two most promising open
threads, left unfinished, are:

- a proper warm-started sweep across d=2,3,4,... for the order-type/convex-
  position family, to see whether the d=3 uptick is the start of a real
  trend or a local bump;
- testing non-local (multi-coordinate, second-order) moves on the M=78
  record, since the LP-oracle diagnostic only rules out first-order local
  improvement, not e.g. simultaneous multi-point swaps.

All code and derivations are in `PermutationProblem/theory_approach/`
(files 00-04 plus their check/optimize scripts).
