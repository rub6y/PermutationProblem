# `rec_AI.py`'s recursive construction: a scaling method that actually works

Plan step 5 asked whether the record generator can be pushed past n=26, and
whether it degrades like Track 1 (`track1_scaling_diagnosis.md`) or holds up.
`records.jsonl` itself turned out to be n=26-only, but the repo already
contains an *unrelated, undocumented* generator that answers this question
directly: `rec_AI.py` / `recurence.py`'s `recurence_construction`.

## What it does

Starting from `S1` (`good_permutation.txt`'s 26 points as a tensor `S1_T`)
and `Sym` (all 6 permutations of $(0,1,2)$, reshaped via `from_perms_to_tensor`
into 3 six-vectors `Sym_T[0..2]`), each recursion step replaces every point
$x$ with 3 new points $3x + \text{Sym\_T}[i]$, $i=0,1,2$ -- a base-3 digit
expansion using the `Sym_T` vectors as digits. One step multiplies the point
count ($M$) by 3 and the resolution by 3.

## It preserves the central hyperplane exactly -- provably, not just empirically

Computed the three digit vectors: `Sym_T[0]=(0,0,1,1,2,2)`, `Sym_T[1]=(1,2,0,2,0,1)`,
`Sym_T[2]=(2,1,2,0,1,0)` -- **each sums to exactly 6**. If $x$ already satisfies
$\sum_k x_k = 3(M-1)$ (on hyperplane $H$, cf. `measure.tex` Conjecture 4.1),
then $\sum_k (3x_k + \text{Sym\_T}[i]_k) = 3\cdot 3(M-1) + 6 = 3(3M-1)$, which
is exactly the hyperplane target for the new size $3M$. So **every point at
every recursion depth sits exactly on $H$, by construction** -- not
approximately, and not something that needs checking numerically. This is a
genuine closed-form, hyperplane-preserving family, unlike
`algebraic_construction_search.py`'s linear family (which sits on $H$ by
construction too, but scores far lower).

## It scales -- and J goes up, not down

Ran `recurence_construction(S1_T, depth)` + `compute_J_fast` (both already
in `rec_AI.py`) for depth 0/1/2:

| depth | M   | J        |
|-------|-----|----------|
| 0     | 26  | 0.493628 |
| 1     | 78  | 0.493956 |
| 2     | 234 | 0.493993 |

J increases monotonically with n here, converging upward toward (but still
below) 0.5 -- the correct direction per the $\mathfrak{S}(n) \to 1/2$
conjecture, and the **opposite** of Track 1's SA-based scaling attempt,
which got worse from n=26 to n=60 (`track1_scaling_diagnosis.md`). A leftover
run already in `rec_AI.py` (`recurence_construction(S1_T, 5)`, M=6318) gives
J=0.493997, consistent with continued slow convergence.

## Where this leaves the plan

This is the single most useful result of the whole investigation for
"pushing past n=26": a provably hyperplane-preserving, genuinely scaling
construction already exists in the repo, was apparently run once (the
leftover depth-5 call and stray print at the bottom of `rec_AI.py`), and
was never written up or connected to the hyperplane conjecture or to
Track 1's regression. Concrete next steps:

- Investigate *why* it works: the per-step gain (0.493628 -> 0.493956,
  +0.000328) is shrinking each step -- worth checking whether it plateaus
  near 0.494 (like the plain SA record) or keeps climbing slowly toward 0.5;
  a few more depths (3, 4 -- M=702, 2106) would show the trend, at the cost
  of $O(M^3)$ compute (already the bottleneck: depth 5's M=6318 run alone
  took several minutes of parallel numba compute).
- `rec_AI.py` has a name suggesting an LLM-assisted construction ("AI") and
  currently runs its expensive depth-5 example as *module-level* code with
  no `if __name__ == "__main__":` guard -- anything that imports the module
  (as this investigation's own quick check did) re-triggers it silently.
  Should be guarded before the module is reused elsewhere.
- Given this scales cleanly, it's a better base for the warm-start
  suggested in `track1_scaling_diagnosis.md` than either
  `hyperplane_seed`'s random construction or `FWPush.md`'s twin-doubling
  blow-up: run hyperplane-constrained local search *on top of* a
  `recurence_construction` seed at each target n, instead of from scratch.
