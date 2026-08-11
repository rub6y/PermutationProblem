# What makes the best `records.jsonl` witness better than the algebraic family?

Follow-up to `track1_scaling_diagnosis.md`, plan step 4. `records.jsonl`'s
best entry (`opt: constraint-preserving`, n=26) scores J=0.494310, slightly
beating `good_permutation.txt` (J=0.493628) and far beating
`algebraic_construction_search.py`'s linear/coprime family (J≈0.2485 at
n=26 -- `results/algebraic_construction_search.json`). Goal: find what
structural feature explains the gap, and whether it's a reusable closed
form.

## Basic checks

- All 6 config columns are genuine permutations of $\{0,\ldots,25\}$, and
  every row sums to exactly 75 = 3·25 -- confirms it sits exactly on the
  central hyperplane $H$ (Conjecture in `measure.tex` §4), same as
  `good_permutation.txt` and every other top record checked in step 1/2.
- **Not a relabeling of `good_permutation.txt`**: after sorting both point
  clouds by coordinate 0 (to align rows canonically), 83 of 156 matrix
  entries still differ. This is a distinct local optimum, not a small
  perturbation of the known one.
- **No pairwise algebraic relation between coordinates**: checked all 15
  coordinate pairs for equality or exact reversal ($\sigma_b = 25-\sigma_a$)
  -- none found. This immediately rules out the kind of structure the
  algebraic family assumes (each of the 3 pairs being a single affine map
  plus its reversal): the winning witness does not pair coordinates that way
  at all.

## Cycle structure of coordinate 0 (as a permutation of $[26]$)

$$
(1\,2)(3\,4\,5)(6\,8)(9\,11\,15\,12)(10\,13)(17\,19\,18)(23\,25\,24)
$$
with 12 fixed points (0, 7, 14, 16, 20, 21, 22, and others). No long cycles,
no arithmetic-progression pattern (contrast with the algebraic family's
$\sigma(i) = (ai+b) \bmod n$, which by construction is close to a single
26-cycle for most coprime $a$). The visible structure is a scatter of small
(2- and 3-element, one 4-element) cycles with no obvious global rule
generating them.

## Conclusion

No closed form was found, and the evidence suggests there probably isn't
one to find at this resolution: the winning configuration looks like the
output of local search that happened to land in a slightly better basin
than `good_permutation.txt`'s, not like a formula. This is consistent with
`opt: constraint-preserving` being a search strategy name, not a
constructive one (unlike `opt: q4-generator`, seen elsewhere in
`records.jsonl`, which *is* a named closed-form family but tops out lower,
same pattern as the algebraic family here).

This changes the practical takeaway for "beating the record": the gain from
0.4936 to 0.4943 came from **more/better local search**, not from
discovering algebraic structure to exploit. It reinforces `track1_scaling_diagnosis.md`'s
recommendation (step 3) rather than adding a competing one: the fastest
path to improving on both fronts is the same fix -- warm-start local search
(hyperplane-constrained hill-climbing/SA) from good known witnesses with a
properly scaled step budget, rather than searching for a closed-form family.
The algebraic-construction line of attack (`algebraic_construction_search.py`)
should be considered a dead end at this resolution unless a fundamentally
different family (not per-pair affine maps) is proposed.
