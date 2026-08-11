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
