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
