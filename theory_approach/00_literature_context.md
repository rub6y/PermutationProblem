# Literature context: how limits like 1/2 get established elsewhere

Goal of this note: before inventing a construction, understand *why* the
fixed-gadget recursion (`rec_AI.py`, see `recurrence_warmstart_results.md`)
provably plateaus at J≈0.493998 — below both 0.5 and the current empirical
record — and what the analogous historical results actually did instead.

## Why a fixed self-similar gadget saturates

Any construction of the form "take a small witness of size m, blow it up
recursively with a fixed correction rule" is, in the language of graph/measure
limits, converging to the **weak\* limit of the IFS attractor measure**. Once
that limit measure exists (and it does here — `rec_AI.py`'s digit expansion
is literally an iterated function system on $H$), $J$ at generation $k$ is
just $J$ of a $k$-step discretization of a *fixed* limit object, and by
continuity of $J$ (measure.tex, Cor. after Prop. 3.9) the sequence has to
converge to $J(\text{limit measure})$ — a single fixed number, not 0.5, unless
the gadget was already optimal. This is exactly what was measured: gains
decay geometrically (ratio ≈ 1/9 per level), the signature of linear
convergence to an interior fixed point of the IFS map, not of an unbounded
climb. **No fixed-size correction rule, iterated forever, can reach a value
strictly above what one step of it already encodes in the limit.** This is a
generic fact about self-similar constructions, not specific to this problem.

## The historical pattern: growing, not self-similar, gadgets

The problems where a conjectured constant *is* reached only in a limit (never
by a fixed finite recursion) are resolved by families indexed by a parameter
$k\to\infty$ where the $k$-th gadget is not a rescaled copy of the $(k-1)$-th
one but a *qualitatively richer* object:

- **Behrend's construction** (largest known 3-AP-free sets, and its
  descendants for cap sets / corners): uses spheres in $\mathbb{Z}^d$ with
  $d = d(k)\to\infty$; density improves because the ambient dimension grows,
  not because a fixed low-dimensional sphere is iterated.
- **Flag algebra "iterated corrections"** (Razborov and successors, for
  Turán-type densities): the extremal construction is a limit object (a
  graphon / permuton) built as an *increasing* sequence of finer and finer
  blow-ups where each stage's local structure is re-optimized against the
  actual objective, not copied verbatim from the previous stage. The
  numerical value approached is exactly $\sup$ over the *whole* limit
  category, and finite truncations only approximate it — there's no claim
  that a single small pattern, blown up, already attains the sup.
- **Quasirandom/pseudorandom permutation and order-type constructions**
  (Goodman–Pollack allowable sequences; Erdős–Szekeres-type order-type
  extremal problems): these get their asymptotic sharpness from an
  *ambient continuous geometric object* (points on a curve, directions on a
  circle) whose combinatorics is exactly computable, and the discrete
  witnesses of every size $n$ are literal finite samples from the *same*
  continuous object — so there is no recursion at all, and no plateau: the
  bound improves by refining the sampling and the choice of the continuous
  parameters (point measure, direction set), which live in an
  infinite-dimensional space unlike the 3-digit IFS gadget.

The third bullet is directly usable here, because of a structural
coincidence: shattering a triple requires all **6** elements of $S_3$ to be
realized among the six coordinates, and $|S_3|=6$. This is *exactly* the
combinatorial datum that classical circular/allowable sequences (points in
convex position, sorted by rotating direction) already compute exactly:
projecting 3 points in general position onto a rotating direction realizes
each of the 6 possible strict orders on exactly one of 6 arcs as the
direction sweeps a full turn. See `01_growing_gadget_amplification.md` for
the construction this suggests.

## What we are *not* doing

We are not trying to re-derive a better upper bound proof, and we are not
questioning the 0.5 conjecture (per instruction). The two files in this
folder pursue only lower-bound constructions and structural necessary
conditions for the maximizer.
