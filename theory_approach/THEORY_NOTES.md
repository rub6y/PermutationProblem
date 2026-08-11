# Theory-approach notes (consolidated)

Merged from 6 separate notes in `theory_approach/` on 2026-08-11 to reduce
file count; content concatenated in the order they were written (00-05).
The `.py` scripts referenced (06-09, check_*, optimize_*) remain as separate
files in this directory. See `/home/ruba/Documents/Math/KNOWLEDGE.md` for the
compact summary.

---


---

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

---

# Growing-gadget construction: convex position / circular sequences

## 1. The coincidence that makes this applicable

Shattering a triple $\{x,y,z\}$ requires the six coordinates
$\sigma_1,\ldots,\sigma_6$ to realize **all six** elements of $S_3$ (the
group of strict total orders on a 3-set) among them, and $|S_3| = 6$. This
is precisely the combinatorial content of the classical fact about points in
convex position, going back to allowable/circular sequences
(Goodman–Pollack 1980, and the earlier "order type" literature building on
Erdős–Szekeres):

> If three points $x,y,z$ are in general position in the plane, and $u_\theta$
> is the unit direction at angle $\theta$, then as $\theta$ sweeps once around
> the circle $[0,2\pi)$, the linear order of $x,y,z$ under "sort by $u_\theta
> \cdot p$" passes through **each of the 6 possible strict total orders of
> $\{x,y,z\}$ exactly once**, each on its own arc, in a fixed cyclic pattern
> (each arc boundary is one adjacent transposition; going halfway around
> reverses the order).

So: take $n$ points $p_1,\ldots,p_n$ in convex position in the plane
(any generic point set on a strictly convex curve, e.g. a circle), and
6 fixed directions $\theta_1,\ldots,\theta_6 \in [0,2\pi)$. Sorting the
points by $u_{\theta_k}\cdot p$ gives a permutation $\sigma_k \in S_n$ for
each $k$. **This is a valid, if restricted, candidate for the extremal
permutation tuple** — restricted because it only explores the sub-family of
6-tuples arising from directional sorts of a convex point set, but this
sub-family is a genuinely different generative mechanism from anything in
`fw/` so far (not a recursion, not an affine/coprime map, not a local-search
output).

## 2. Exact formula for when a triple is shattered

Put the $n$ points on the unit circle at angles $\varphi_1,\ldots,\varphi_n$
(WLOG — any strictly convex curve is affinely/topologically equivalent for
this combinatorics as long as no two chords are parallel, i.e. genericity).
For two points at angles $\varphi_p,\varphi_q$:
$$
p - q = 2\sin\!\Big(\tfrac{\varphi_p-\varphi_q}{2}\Big)\,
\Big(-\sin\tfrac{\varphi_p+\varphi_q}{2},\ \cos\tfrac{\varphi_p+\varphi_q}{2}\Big),
$$
so the direction $u_\theta$ is perpendicular to $p-q$ (the critical direction
at which their relative order along $u_\theta$ flips) exactly when
$$
\theta \equiv \frac{\varphi_p+\varphi_q}{2} \pmod \pi .
$$
Call $m_{pq} := \tfrac{\varphi_p+\varphi_q}{2} \bmod \pi$ the **midpoint
angle** of the pair. For a triple $x,y,z$ the three midpoint angles
$m_{xy}, m_{xz}, m_{yz} \in [0,\pi)$ each lift to two antipodal critical
points in $[0,2\pi)$ ($m$ and $m+\pi$), giving 6 points that cut the circle
into 6 arcs — and, by the classical fact above, these are exactly the 6
arcs realizing the 6 elements of $S_3$ in cyclic order.

**Shattering condition.** The triple $\{x,y,z\}$ is shattered by
$(\theta_1,\ldots,\theta_6)$ iff each of the 6 arcs cut by
$\{m_{xy}, m_{xy}+\pi, m_{xz}, m_{xz}+\pi, m_{yz}, m_{yz}+\pi\}$ contains
**exactly one** $\theta_k$ (a "rainbow" hit).

Only the three points $m_{xy}, m_{xz}, m_{yz}$ on the circle of
circumference $\pi$ (call it $\mathbb{R}/\pi\mathbb{Z}$) matter — call this
triple of points the triple's **midpoint configuration**. So:
$$
J(\text{this construction}) =
\Pr_{x,y,z \sim \text{point measure}}\big[\, \Theta \text{ is rainbow for the midpoint configuration of } x,y,z \,\big]
$$
where $\Theta = \{\theta_1,\ldots,\theta_6\} \bmod \pi$ lifted to the 6-arc
picture on $[0,2\pi)$, and the probability is over 3 iid draws from
whatever measure $\rho$ on the circle governs the point positions (the
$n\to\infty$ limit of the point set — this $\rho$ is the actual free
parameter, analogous to $\mu \in \mathcal{A}$, but living in the
much smaller space of probability measures on a circle).

This turns the optimization over $\mu \in \mathcal{A}$ (probability measures
on $[0,1]^6$) into an optimization over the pair $(\rho, \Theta)$: a measure
on a circle, and 6 directions. Far smaller search space, but *exactly
computable* rather than combinatorially searched.

## 3. Symmetric special case: uniform $\rho$, equally spaced $\Theta$

Take $\rho = $ uniform measure on the circle, $\theta_k = (k-1)\cdot 60°$.
By rotation invariance, WLOG $\varphi_x = 0$ and $\varphi_y,\varphi_z$
uniform. This reduces $J$ to a finite-dimensional integral
$$
J_0 = \frac{1}{(2\pi)^2}\int_0^{2\pi}\!\!\int_0^{2\pi}
\mathbb{1}\big[\Theta \text{ rainbow for } (0,\varphi_y,\varphi_z)\big]
\, d\varphi_y\, d\varphi_z ,
$$
which is piecewise constant in $(\varphi_y,\varphi_z)$ (a finite hyperplane
arrangement), hence computable exactly by symbolic case analysis or to
arbitrary precision numerically. This is `check_circular_construction.py`
in this folder — see its output for $J_0$.

**This single symmetric configuration is not expected to already be near
0.5** (six equally spaced directions and a uniform circle is the "most
naive" choice, playing the role of the trivial/no-correction baseline). It
matters only as the base case of the growing-gadget family below and as a
sanity check that the geometric reformulation reproduces sensible $J$ values.

**Checked** (`check_circular_construction.py`): the discrete construction
(golden-angle-spaced points on a circle, sorted by 6 evenly spaced
directions, evaluated with `fw.shattering.count_shattered_triples`) and the
independent continuous Monte Carlo estimate (direct sampling of random
angle triples against the midpoint/arc rainbow criterion) agree to 3+
significant figures and both converge to
$$
J_0 = \frac{2}{9} \approx 0.22222\ldots
$$
as $n$ (resp. sample count) grows (measured: $n=30,60,120,240$ give
$0.2209, 0.2217, 0.2221, 0.2222$; 2,000,000-sample MC gives $0.2226$). This
**validates the geometric formula** (independent derivations match) but
confirms the symmetric baseline is far from competitive on its own — it
must be substantially improved by optimizing $(\rho,\Theta)$ (step 2 below)
before this family says anything about closing the gap to 0.5, and even
then may plateau (exactly the failure mode this whole exercise is trying to
avoid) unless $d>2$ is genuinely needed.

## 4. Where the "growing gadget" enters: higher-dimensional order types

The plane ($d=2$) restricts us to 6 *directions on a circle* — a
1-parameter family of order types, which is far too rigid to be optimal (the
cyclic structure forces strong correlations between which arcs are
adjacent). The natural generalization, matching Goodman–Pollack's own
extension of allowable sequences and the theory of **arrangements of great
circles / order types in $\mathbb{R}^d$**: put the $n$ points in convex
position on the moment curve (or any generic curve) in $\mathbb{R}^d$ for
$d\to\infty$, and let $\theta_1,\ldots,\theta_6 \in S^{d-1}$ be 6 *generic
unit vectors* (linear functionals), not constrained to lie in a single
rotating plane. Sorting by $u_{\theta_k}\cdot p$ still gives 6 permutations
$\sigma_k$; but the combinatorics of which of $S_3$'s six orders appear for
a given triple, as a function of $(\theta_1,\ldots,\theta_6)$, is governed
by a **hyperplane arrangement in $\mathbb{R}^d$** determined by the triple
(three points still only span a 2-plane, so the relevant "critical set" for
each triple is still a union of at most 3 great subspheres of $S^{d-1}$ —
but now 6 fixed points on $S^{d-1}$ have $d-1$ continuous degrees of freedom
each to avoid unwanted alignments across *different* triples simultaneously,
something impossible on the circle where all triples' critical sets live on
the same 1-dimensional object and interact rigidly).

This is the actual "growing gadget" sequence to construct: for each $d$,
optimize $(\rho_d, \Theta_d)$ — a measure on $S^{d-1}$ and 6 points on it —
over this restricted but exactly-parametrized family, and ask whether
$\sup_d J(\rho_d,\Theta_d) \to \tfrac12$. This is a genuinely different
conjecture from Conjecture 4.1 in `measure.tex` (hyperplane concentration)
and does not require it; it is a *constructive* question, answerable
independently by exhibiting good $(\rho_d,\Theta_d)$ for increasing $d$.

## 5. Concrete next steps (falsifiable, in order of effort)

1. ~~Compute $J_0$ for the $d=2$ symmetric baseline exactly (Section 3);
   cross check against `fw.objective.compute_J` on a discretized instance.~~
   **Done** — see Section 3: $J_0 = 2/9$, confirmed by two independent
   methods. Formula validated.
2. Within $d=2$, numerically optimize over $(\rho, \Theta)$ jointly
   (non-uniform point measure, non-equally-spaced directions) to see how
   far above $J_0$ a $d=2$ optimum can go — this bounds how much is lost by
   fixing $d=2$ before spending effort on $d>2$.
3. If $d=2$'s ceiling (after optimizing $\rho,\Theta$) is well below the
   current record (0.495195), the family is not competitive at low $d$ and
   the value of $d$ needed to close the gap becomes the real question —
   test $d=3$ next (still cheap: 6 points on $S^2$, an optimizable point
   measure on a curve in $\mathbb{R}^3$) rather than jumping to large $d$.
4. Only if $d=2,3$ show a clear increasing trend in $\sup J$ as $d$ grows is
   it worth the investment of coding the general-$d$ hyperplane-arrangement
   machinery and searching larger $d$.

If step 1 or 2 shows the construction cannot beat ~0.49 even after full
optimization at $d=2$, this specific geometric mechanism should be reported
as falsified for low $d$ and the growing-$d$ question left open rather than
pursued further without evidence of an increasing trend.

## 6. Step 2 result: $d=2$ optimized ceiling is $\approx 0.2498$, not competitive

`optimize_circular_construction.py` runs simulated annealing directly on the
exact discrete objective (`fw.shattering.count_shattered_triples`), jointly
over all $n$ point-angles $\varphi_i$ and the 6 direction angles
$\theta_1,\ldots,\theta_6$ (both free, not constrained to be evenly spaced
or uniform) — i.e. a full joint optimization of $(\rho,\Theta)$ in the
discretized version of this family. Runs at $n=60, 80, 100$ (2 independent
seeds each, annealed from the symmetric $J_0=2/9\approx0.222$ baseline)
all converge tightly to the same value regardless of $n$ or seed:

| $n$ | best $J$ (over seeds) |
|---|---|
| 60  | 0.24972 |
| 80  | 0.24983 |
| 100 | 0.24982 |

**This is a real ceiling, not an under-optimized baseline**: independent
seeds at three different resolutions all land within $10^{-4}$ of $0.2498$,
and it is essentially the same number found by the *entirely different*
reflection-pair torus family in `03_torus_reflection_pairs.md`
($\approx 0.225$–$0.2485$ after its own optimization). Two structurally
unrelated $d=2$/1-parameter-curve constructions — one from rotating
projections of a convex point set, one from circle-doubling with reflection
pairs — independently cap out at essentially the same $\approx 0.25$, which
is suggestive of a genuine structural limitation of *any* measure supported
on a 1-dimensional curve (or a small number of curves) in $[0,1]^6$, not an
artifact of either specific parametrization.

**Conclusion on $d=2$: falsified as a competitive family.** $0.2498 \ll
0.495195$ (the current record) even after fully optimizing both the point
measure and the projection directions — optimizing $(\rho,\Theta)$ within
$d=2$ closes almost none of the gap to the record, let alone to $0.5$. Per
the plan in §5, the next test is whether $d=3$ (6 directions on $S^2$, point
measure on a curve in $\mathbb{R}^3$) breaks through this $\approx0.25$
plateau or merely reproduces it — if $d=3$ also caps near $0.25$, that would
be strong evidence the whole "convex position + few generic directions"
mechanism is capped independent of $d$, and growing $d$ is not the right
lever; if $d=3$ shows real improvement, it supports the growing-gadget
hypothesis and justifies pushing to larger $d$. ## 7. $d=3$ result: a small but real improvement over the $d=2$ ceiling

`optimize_sphere_construction.py` runs the same joint annealing (now over
$n$ points on $S^2$ and 6 direction vectors in $\mathbb{R}^3$, initialized
from a Fibonacci-sphere point set and the octahedron's 6 axis directions as
the natural "evenly spaced" $d=3$ baseline).

- **Baseline** (no optimization, octahedron directions): $n=40\to
  J=0.2483$, $n=70\to J=0.2491$ — already matching the *fully optimized*
  $d=2$ ceiling ($\approx 0.2498$) without any search at all. This alone is
  informative: the naive $d=3$ configuration is roughly as good as the best
  $d=2$ could do after extensive annealing.
- **After annealing**: $n=40\to J=0.2491$ (5,000 steps, barely above
  baseline); $n=70\to J=0.2568$ (60,000 steps) — a **real, clearly
  above-noise improvement over the $d=2$ ceiling** (+0.007, i.e. about
  3% relative), and the anneal trace shows $J$ still climbing at the end of
  the run (0.248 at step 36k → 0.257 at step 60k, not yet plateaued),
  unlike the $d=2$ runs which flattened out cleanly.

**Interpretation.** This is a real but small data point in favor of the
growing-$d$ hypothesis (§4): going from $d=2$ to $d=3$ did buy a genuine
improvement, not just noise, and the $n=70$ run hadn't converged yet
(more annealing steps or a bigger $n$ might push it further). But the
improvement is far too small, extrapolated naively, to reach anywhere near
$0.5$ without either (a) much larger $d$ than is computationally convenient
to test this way (each $d$ needs its own annealing run, and the per-step
cost already dominates at $n=70,d=3$), or (b) some non-obvious jump in
behavior at larger $d$ that a two-point ($d=2,3$) trend can't reveal.

**Honest conclusion:** this line is not yet falsified the way $d=2$ alone
was, but it is also not close to being a competitive construction at the
$d$ values that are cheap to test. Continuing to $d=4,5,\ldots$ by hand,
one slow annealing run at a time, is a poor use of effort relative to its
information yield — a $d=4$ point alone would not distinguish "slowly
climbing to 0.5" from "climbing to some other plateau below 0.4," which is
exactly the ambiguity a single additional data point cannot resolve. If
this direction is pursued further, the right next step is not "try $d=4$"
but building a version that can sweep $d$ semi-automatically (reusing the
annealed solution at $d$ to warm-start $d+1$, and tracking the *sequence*
of ceilings rather than one-off points) so the trend itself, not a single
new number, is the object being measured.

---

# Perturbative / calculus-of-variations approach

## 1. Setup

`measure.tex` (Prop. "linear necessary condition for maximizers", and its
extension via multi-marginal duality, §"Closing remarks") already gives the
right first-order machinery: if $\mu$ maximizes $\mathcal{J}$ over
$\mathcal{A}$, then $\mu$ also maximizes the *linear* functional
$$
\nu \mapsto \mathcal{J}(\mu,\mu,\nu) = \int_{[0,1]^6} g_\mu \, d\nu, \qquad
g_\mu(\mathbf{z}) = \int 1_{\mathcal{D}}(\mathbf{x},\mathbf{y},\mathbf{z})\, d\mu(\mathbf{x})\,d\mu(\mathbf{y}),
$$
over $\mathcal{A}$, and there exist potentials $u_1,\ldots,u_6$ with
$g_\mu(\mathbf z) \le \sum_k u_k(z_k)$ everywhere and equality $\mu$-a.e.
(Prop. "duality necessary condition"). $g_\mu$ is exactly (up to the
constant factor already tracked in `fw.objective`) the Frank–Wolfe gradient.

This gives a genuine, checkable **first-order optimality test** for any
candidate $\mu$ (in particular the current record, M=78, J=0.495195):

> $\mu$ can be a local max only if there is **no point** $\mathbf z$ with
> $g_\mu(\mathbf z) > \sum_k u_k(z_k)$ for the potentials realized on
> $\mathrm{supp}(\mu)$ — equivalently, only if the discrete LP oracle
> (`fw/lp_oracle.py`), run against the gradient at $\mu$, cannot find any
> point outperforming the current support.

This is **not new machinery** — it is precisely what one step of
Frank–Wolfe already checks — but it has not been used as a *diagnostic
question in its own right*: "is the current record a stationary point of
$\mathcal{J}$ restricted to $\mathcal{A}$ at all, or is there slack left on
the table by the search procedures that produced it?" `records_best_structure.md`
and `NEXT_recurrence_scaling_plan.md` treat the current record purely as a
search *output*; nobody has reported running one full LP-oracle gradient
step *from* it to check for a strictly ascending direction still available
within $\mathcal{A}$.

## 2. The concrete test to run (numerics-to-falsify, per plan)

1. Load the M=78 record measure $\mu^*$ (uniform weight $1/78$ on its 78
   support points, from `recurrence_warmstart_results.md`'s witness).
2. Compute the exact gradient $g_{\mu^*}(\mathbf z)$ at every candidate grid
   point $\mathbf z$ reachable at this resolution (`fw.objective.compute_gradient`,
   reusing the existing candidate pool machinery in `fw/candidates.py`).
3. Solve the assignment/transportation LP oracle
   (`fw/lp_oracle.py::solve_lp_oracle`) against this gradient to find the
   $\nu \in \mathcal{A}_n$ maximizing $\langle g_{\mu^*}, \nu\rangle$.
4. Compare $\langle g_{\mu^*}, \nu \rangle$ against $\langle g_{\mu^*},
   \mu^* \rangle = 3\,\mathcal{J}(\mu^*)$ (self-consistency: this is what
   the complementary-slackness equality in Prop. "duality necessary
   condition" says should hold at a true maximizer). A **strictly larger**
   value means a genuine ascent direction exists at the *first order* level
   and a full Frank–Wolfe/line-search step from $\mu^*$ should be tried
   before trusting $\mu^*$ as any kind of local optimum — this is separate
   from (and cheaper than) the basin-hopping search already tried, because
   it uses the exact LP rather than random perturbations.

This is a falsification test in the strict sense requested: either it finds
slack (actionable — push further from $\mu^*$ using the *exact* ascent
direction rather than random swaps) or it confirms $\mu^*$ is already a
first-order stationary point of $\mathcal{J}|_\mathcal{A}$ at this
resolution (informative — it means further gains, if any, require changing
$n$/resolution, not more search at fixed $n$, consistent with
`track1_scaling_diagnosis.md`'s finding that resolution, not search budget,
was the bottleneck for track 1).

**Result** (`check_record_stationarity.py`, run against the M=78,
J=0.495195 record): loaded the record's 6 permutations, built the exact
gradient over a 56,114-point candidate pool (the record's own support, every
single-coordinate variant of every support point, and 20,000 random points —
the same pool machinery `fw/candidates.py` already uses for Frank–Wolfe),
and solved the LP oracle against it.
$$
\langle g_{\mu^*}, \mu^*\rangle = 3\,\mathcal{J}(\mu^*) = 1.48558641,
\qquad
\langle g_{\mu^*}, \nu_{\text{LP}}\rangle = 1.48558641,
$$
agreeing to $4\times10^{-16}$ (floating-point noise). **No ascent
direction exists in this candidate pool.** $\mu^*$ is a genuine first-order
stationary point of $\mathcal{J}$ restricted to $\mathcal{A}_{78}$ against
every single-coordinate perturbation and 20,000 random resamplings — the
basin-hopping search that produced it had, in fact, already exhausted the
easy local structure at this resolution. This confirms (rather than merely
suggests) that further gains at $M=78$ specifically require either a
genuinely non-local move (multi-coordinate simultaneous swaps outside the
tested pool) or moving to larger $n$/different structure entirely — it
rules out "just search harder at M=78" as a productive next step.

## 3. Why this alone cannot resolve "does $J\to 1/2$"

Even a perfect confirmation that $\mu^*$ is a strict local (even global, at
fixed $n$) maximizer of $\mathcal{J}_n$ says nothing about the limit
$n\to\infty$: $\mathcal{J}_n(\mu^*) \le \mathcal{J}(\mu^*) \le \sup_\mu
\mathcal{J}(\mu)$, and none of these inequalities need be tight as
$n\to\infty$ just because $\mu^*$ is optimal at its own resolution — this
is exactly the failure mode already observed empirically (`rec_construction_scaling.md`):
locally-refined finite witnesses plateau below 0.5 even though they are
genuine (near-)maximizers at their own $n$. The real question a
perturbative approach needs to answer is: as $n \to \infty$ along a
*sequence* of first-order-stationary $\mu_n^*$, does
$\liminf \mathcal{J}(\mu_n^*)$ actually climb toward $1/2$, or does it
converge to something below (mirroring the IFS-recursion plateau, just with
a possibly larger limit)? Answering that requires either

- **(a)** an explicit ascent construction that provably increases
  $\mathcal{J}$ without bound as a parameter $\to\infty$ (this is what
  `01_growing_gadget_amplification.md` attempts, geometrically), or
- **(b)** a genuine variational/PDE-style argument bounding the *rate* at
  which the discretization gap $\mathcal J(\mu) - \mathcal J_n(\mu)$
  vanishes, combined with a uniform (in $n$) lower bound on
  $\max_{\mu\in\mathcal A_n}\mathcal J_n(\mu)$ that grows to $1/2$ — which
  would need new input beyond what the LP-oracle diagnostic alone provides.

So: treat this file's test as a **cheap, immediate sanity/diagnostic step**
to run before investing further into either the growing-gadget or any other
construction — not as a standalone route to the limit.

## 4. A second, more ambitious first-order idea (harder, flagged as open)

If the codimension-1 concentration conjecture is assumed (user's stated
belief, and it is well supported empirically), it is worth asking the
*restricted* variational problem: over $\mu \in \mathcal{A}$ with
$\mathrm{supp}(\mu)\subset H$, does the Fréchet derivative of
$\mathcal{J}|_{\mathcal A \cap \{\mathrm{supp}\subset H\}}$ admit a
one-parameter ascent family $\mu_t$, $t\to\infty$, along which $J(\mu_t)\to
1/2$? Because $H \cap [0,1]^6$ is a 5-dimensional polytope (not all of
$\mathbb{R}^6$), and the marginal constraints restricted to $H$ are still
linear, this is a lower-dimensional but structurally identical problem to
the original one (same cubic functional, same convex marginal polytope,
codimension 1). At present nothing suggests the restriction to $H$ makes
the variational problem tractable in closed form; this is recorded as an
open direction rather than pursued now, since the falsification test in
§2 is the higher-priority, cheaper next step.

---

# Integrals over a winding-line manifold: the "reflection-pair" family

This follows up on the user's prompt to look at manifolds that already
showed up in the results (`records_best_structure.md`'s observation that the
`algebraic_construction_search.py` / `q4-generator` family pairs each
coordinate with an affine map plus its exact reversal, $\sigma_b = n -
\sigma_a$) and to try computing exact integrals rather than more discrete
search.

## 1. The continuum limit of the reversal-pair family

The discrete family used coordinate pairs $(\sigma_a, n-\sigma_a)$ built
from an affine/coprime generator. Its continuum ($n\to\infty$) limit is the
pushforward measure $\mu_\beta \in \mathcal{A}$ of Lebesgue measure on
$[0,1)$ under
$$
t \ \longmapsto\ \big(t,\ 1-t,\ \{\beta_2 t\},\ 1-\{\beta_2 t\},\
\{\beta_3 t\},\ 1-\{\beta_3 t\}\big),
$$
where $\{\cdot\}$ is fractional part and $\beta_2,\beta_3 \in \mathbb{R}$
are free slopes ($\beta_1=1$ after rescaling time). This is literally a line
of slope $(1,\beta_2,\beta_3)$ wound onto the 3-torus and then doubled by
reflection into 6 coordinates — a genuine (1-dimensional) sub-manifold of
$[0,1]^6$, exactly matching what the discrete search was sampling from at
finite $n$. Each coordinate's marginal is Lebesgue on $[0,1)$ (true for any
real $\beta$, not just integers), so $\mu_\beta \in \mathcal{A}$ for every
$(\beta_2,\beta_3)$, and it lies on $H$ automatically: each pair sums to
exactly 1, so the total is exactly 3 for every $t$.

## 2. Exact reduction: this 6-dimensional family is really 3-dimensional

Reflection $x\mapsto 1-x$ exactly reverses the strict order of any triple:
if `order_code` (the project's canonical $S_3$-labeling,
`fw/shattering.py`) assigns code $c$ to $(a,b,c')$, it assigns code $5-c$ to
$(1-a,1-b,1-c')$ — checked directly from the case tree, giving the
fixed-point-free involution pairing $\{0,5\},\{1,4\},\{2,3\}$ on the six
codes. Consequently, for *any* triple $t_x,t_y,t_z$, coordinates $(1,2)$
automatically realize one whole pair $\{c,5-c\}$ of the 6 needed codes,
whatever $c$ is; likewise pairs $(3,4)$ and $(5,6)$. Shattering — needing
all 6 codes to appear — therefore holds **iff the three "pair classes"**
$$
\mathrm{cls}(a,b,c') := \min\big(\mathrm{code}(a,b,c'),\,5-\mathrm{code}(a,b,c')\big) \in \{0,1,2\}
$$
computed from $(t_x,t_y,t_z)$, $(\{\beta_2 t_x\},\{\beta_2 t_y\},\{\beta_2
t_z\})$, and $(\{\beta_3 t_x\},\{\beta_3 t_y\},\{\beta_3 t_z\})$
respectively, are **pairwise distinct** — i.e. hit all of $\{0,1,2\}$
exactly once. So
$$
\mathcal{J}(\mu_\beta) = \Pr_{t_x,t_y,t_z \sim \mathrm{Unif}[0,1)}
\Big[\ \mathrm{cls}(1;t_x,t_y,t_z),\ \mathrm{cls}(\beta_2;\cdot),\
\mathrm{cls}(\beta_3;\cdot)\ \text{ are pairwise distinct}\ \Big],
$$
an exactly-defined probability depending on only 2 real parameters
$(\beta_2,\beta_3)$ — a genuine reduction of a 6-dimensional variational
problem to a 2-parameter one, via the reflection symmetry alone (no search
needed to find this structure; it falls out of the pairing algebra).

This also explains, retroactively, *why* the discrete version of this
family was reported as a dead end (`records_best_structure.md`): it isn't
that the discrete search under-explored it, it's that the entire family —
for every real choice of $(\beta_2,\beta_3)$ — reduces to a 3-way
equidistribution question about a single circle map $t\mapsto\{\beta t\}$,
which has no extra parameters left to exploit once $\beta_2,\beta_3$ are
fixed. There is no hidden richer regime to search into.

## 3. Numerical evaluation (`check_torus_reflection_family.py`)

Monte Carlo evaluation of $\mathcal{J}(\mu_\beta)$ (400k–600k samples,
using the project's own `order_code`) over integer slopes matching the
discrete family (e.g. $\beta=(2,3),(3,5),(5,7)$) gives values in
$0.12$–$0.21$; a further random search over $60$ real slope pairs in
$[1.2,40]^2$ tops out at $J \approx 0.2255$ (at $\beta\approx(25.0,
36.8)$), i.e. **this whole continuous family caps out well below both the
$n=26$ discrete instance's reported value (0.2485) and, by a wide margin,
the current record (0.495195)**. The discrete family likely exceeds the
continuum cap slightly because of finite-$n$ boundary/discretization
effects (matching the general pattern already seen with the digit-recursion
family: finite instances can transiently beat their own continuum limit by
a small amount before settling).

**Conclusion: confirmed dead end, now via an exact reduction rather than
just search.** No amount of tuning $(\beta_2,\beta_3)$ can bring this
manifold-family close to 0.5; the mechanism (pure reflection-pairing) throws
away too much of the 6-way combinatorial freedom by construction (it forces
each of the 3 "class-choices" to come from literally the *same* map $t
\mapsto \{\beta t\}$ evaluated at 3 different points, rather than allowing
each of the 6 coordinates to depend on genuinely different structure — this
is the real content of why a richer, non-reflection-paired family, such as
`01_growing_gadget_amplification.md`'s convex-position construction, is a
better use of further effort than more slopes in this family).

## 4. Takeaway for prioritization

This closes off a line the previous computational work had already flagged
as unpromising, but is worth having on record as a *proved* dead end (exact
reduction + numerics) rather than an *empirically suspected* one — the
30-second closed-form reduction here is cheap insurance against
re-attempting it later with bigger search budgets. Effort should go to
`01_growing_gadget_amplification.md`'s $d>2$ direction, or a genuinely
non-reflection-symmetric manifold family, rather than more slopes here.

---

# A cheap negative control: independence actively hurts

Every construction tried so far (recursion, algebraic/reflection-pair,
circular, sphere) imposes strong *correlation* between the 6 coordinates
(all forced onto or near the hyperplane $H$). Worth checking the opposite
extreme cheaply before pursuing more structured ideas: what if the 6
permutations are made deliberately *uncorrelated*?

**Theory.** If the 6 order-types of a random triple were independent and
uniform over $S_3$ (as they are, asymptotically, for genuinely unrelated
permutations), $\mathcal J \to 6!/6^6 = 720/46656 \approx 0.015432$ — the
probability that 6 iid uniform draws from a 6-element set happen to be a
bijection.

**Check** (`check_independence_baseline.py`): iid random permutations of
$[n]$ give $J \in [0.009, 0.017]$ for $n=26,60,100$, tracking the
theoretical limit closely. A structurally different "pseudorandom" family
built from discrete exponentiation mod a prime (multiplicative, not
affine — a genuinely different generator from every algebraic family tried
in `03`) gives $J=0.0078$ at $n=100$ — same ballpark, no better.

**Conclusion.** Independence is not a viable strategy at any scale; it's
$16$–$30\times$ worse than even the weak circular baseline ($0.22$) and
$\sim 60\times$ worse than the record ($0.495$). This is a cheap but
useful negative control: it confirms that *all* of the gain in every
competitive construction comes from forcing strong coordinate correlation
(concentration near $H$), not from cleverly diversifying the 6
permutations — reinforcing why Conjecture "codim-1 concentration"
(`measure.tex` §4) is the right thing to keep believing and building
constructions around, and ruling out "more pseudorandom generators" as a
line of attack.

---

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
