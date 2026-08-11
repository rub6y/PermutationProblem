"""Symmetry group of J (measure.tex reformulation).

CORRECTED after a numerical check falsified the first draft of this module:
reflecting a *proper subset* of the 6 axes does NOT preserve the shattering
predicate. Reflecting one coordinate composes only that coordinate's
order-code with the order-reversing permutation rho of S_3, while leaving
the other five order-codes alone; since the six codes of a shattered triple
range over all of S_3, this pointwise (per-slot) recoding is not a global
post-composition and need not stay injective -- verified to break shattering
on real triples from good_permutation.txt (see conversation).

What DOES provably preserve the shattering predicate (confirmed numerically
via fw.shattering.count_shattered_triples on good_permutation.txt, count
unchanged in every case):
  - any permutation of the 6 coordinate axes (relabels which S_3 element
    sits in which coordinate -- precomposition of the [6] -> S_3 bijection
    by a bijection, still a bijection);
  - reflecting ALL 6 axes simultaneously, x -> (n - 1) - x elementwise
    (post-composes the [6] -> S_3 bijection with the single fixed
    order-reversing permutation rho, applied uniformly, still a bijection).

So the true symmetry group is G = S_6 x Z_2 (order 6! * 2 = 1440), not the
full hyperoctahedral group B_6 (order 46080) originally claimed.

This module builds orbit-averaged measures under G (or its S_6-only
subgroup): for a measure mu, mu_sym = (1/|G|) * sum_g g_* mu is again in A_n
(each g individually preserves uniform marginals), and if mu were the unique
maximizer of J, mu_sym would equal mu.
"""

import itertools

import numpy as np

from fw.measure import Measure

N_COORDS = 6


def reflection_transforms():
    """The only two axis-reflection patterns that preserve shattering: no
    reflection, or reflecting all 6 axes at once (see module docstring --
    reflecting a proper subset breaks the shattering predicate)."""
    return [None, np.ones(N_COORDS, dtype=bool)]


def permutation_transforms():
    """All 6! = 720 permutations of the 6 coordinate axes, as index arrays
    usable for `points[:, perm]`."""
    return [np.array(p, dtype=np.int64) for p in itertools.permutations(range(N_COORDS))]


def apply_transform(points, n, perm=None, reflect=None):
    """Apply one B_6 element to an (M, 6) int array of grid points: first
    permute axes by `perm` (or identity if None), then reflect the axes
    marked True in `reflect` (or none if None) via x -> (n - 1) - x."""
    out = points if perm is None else points[:, perm]
    if reflect is not None and reflect.any():
        out = out.copy()
        out[:, reflect] = (n - 1) - out[:, reflect]
    return out


def orbit_average(measure, perms=None, reflects=None):
    """mu_sym = (1/|G'|) sum_{g in G'} g_* mu for G' = the product of the
    given permutation list and reflection list (identity-only list if either
    is omitted). Duplicate grid points across the orbit have their weights
    summed rather than kept as separate atoms."""
    perms = perms if perms is not None else [None]
    reflects = reflects if reflects is not None else [None]
    n = measure.n
    combined = {}
    group_size = len(perms) * len(reflects)
    for perm in perms:
        for reflect in reflects:
            transformed = apply_transform(measure.points, n, perm, reflect)
            for point, weight in zip(map(tuple, transformed), measure.weights):
                combined[point] = combined.get(point, 0.0) + weight / group_size
    points = np.array(list(combined.keys()), dtype=np.int64)
    weights = np.array(list(combined.values()), dtype=np.float64)
    return Measure(points=points, weights=weights, n=n)


def global_reflection_orbit_average(measure):
    """Symmetrize under the 2-element global-reflection subgroup {id, flip
    all 6 axes} only (no axis permutation)."""
    return orbit_average(measure, reflects=reflection_transforms())


def permutation_orbit_average(measure):
    """Symmetrize under the 720-element axis-permutation subgroup S_6 only
    (no reflection)."""
    return orbit_average(measure, perms=permutation_transforms())


def full_orbit_average(measure):
    """Symmetrize under the full 1440-element group G = S_6 x Z_2."""
    return orbit_average(measure, perms=permutation_transforms(), reflects=reflection_transforms())
