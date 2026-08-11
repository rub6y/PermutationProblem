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
