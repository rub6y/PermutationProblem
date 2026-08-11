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
