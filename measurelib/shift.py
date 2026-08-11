"""Marginal-preserving moves: A_n -> A_n. The "shift a measure to another
one" primitives.

`interpolate` is the coarse move (jump toward an arbitrary target measure).
`cycle_move` / `two_coordinate_cycle` / `random_two_coordinate_move` are the
fine, *local* move: shifting weight around a small alternating cycle of grid
points changes the measure while leaving every one of the 6 marginals
exactly fixed, for ANY weighted measure -- not just the equal-weight,
one-point-per-row permutation measures fw/swap_search.py and
fw/hyperplane_search.py are restricted to. This is the elementary step
gradient-descent/annealing style search over general measures needs.

Why a cycle preserves marginals: pick 2 coordinates (a, b) and hold the
other 4 coordinates fixed at some tuple `rest`. Within that single fiber,
consider a cyclic sequence of grid points alternating between "same
a-value, next b-value" and "same b-value, next a-value" steps, e.g. for
values a0,a1,...,a{k-1} and b0,b1,...,b{k-1}:
  (a0,b0) -> (a0,b1) -> (a1,b1) -> (a1,b2) -> ... -> (a{k-1},b0) -> (a0,b0).
Assigning weight delta +eps/-eps alternately around this closed walk keeps
every marginal fixed: coordinates outside {a,b} are constant over the whole
cycle, so their marginal bucket's net change is the sum of the deltas, which
is 0 (equal counts of +eps and -eps). For coordinate a (resp. b), each value
a_i (resp. b_i) is shared by exactly the two points adjacent to it in the
cycle, one with each sign, so their contributions cancel pairwise.

Known limitation: this only covers moves confined to a single 4-coordinate
fiber (2 coordinates varying at a time). A general cycle spanning more than
2 coordinates at once (a "hypergraph augmenting cycle" over the full support)
would cover strictly more moves but needs real cycle-finding and is left for
a later round once/if the 2-coordinate move proves too restrictive for the
search that consumes it.
"""

import numpy as np

from measurelib.measure import N_COORDS, Measure

_TOL = 1e-12


def interpolate(mu_a, mu_b, t):
    """(1 - t) * mu_a + t * mu_b, merging matching support points rather than
    duplicating atoms. The coarse "shift toward a target measure" tool."""
    combined = {}
    for point, weight in zip(map(tuple, mu_a.points), mu_a.weights):
        combined[point] = combined.get(point, 0.0) + (1.0 - t) * weight
    for point, weight in zip(map(tuple, mu_b.points), mu_b.weights):
        combined[point] = combined.get(point, 0.0) + t * weight
    points = np.array(list(combined.keys()), dtype=np.int64)
    weights = np.array(list(combined.values()), dtype=np.float64)
    keep = weights > _TOL
    return Measure(points=points[keep], weights=weights[keep], n=mu_a.n)


def two_coordinate_cycle(rest_coords, rest_values, coord_a, coord_b, a_seq, b_seq):
    """Build the alternating cycle described in the module docstring: fixed
    `rest_values` on `rest_coords`, varying `coord_a` through `a_seq` and
    `coord_b` through `b_seq` (both length k >= 2, no repeated values).
    Returns (points (2k, 6) int64, signs (2k,) int in {+1, -1})."""
    k = len(a_seq)
    if len(b_seq) != k or k < 2:
        raise ValueError("a_seq and b_seq must have equal length >= 2")
    if len(set(a_seq)) != k or len(set(b_seq)) != k:
        raise ValueError("a_seq and b_seq must not repeat values")

    points = np.empty((2 * k, N_COORDS), dtype=np.int64)
    signs = np.empty(2 * k, dtype=np.int64)
    for coord, value in zip(rest_coords, rest_values):
        points[:, coord] = value

    for i in range(k):
        # "same a, b_i -> b_{i+1}" step, then "same b, a_i -> a_{i+1}" step
        points[2 * i, coord_a] = a_seq[i]
        points[2 * i, coord_b] = b_seq[i]

        points[2 * i + 1, coord_a] = a_seq[i]
        points[2 * i + 1, coord_b] = b_seq[(i + 1) % k]

    # Flat alternation by position (not by i): each a-value occupies two
    # adjacent flat positions 2i, 2i+1 and each b-value occupies two adjacent
    # flat positions 2i-1, 2i (mod 2k) -- alternating +1/-1 by flat position
    # makes both pairs cancel regardless of k (see module docstring).
    signs[0::2] = 1
    signs[1::2] = -1

    return points, signs


def cycle_move(mu, points, signs, epsilon):
    """Apply weight delta = epsilon * sign at each cycle point (points, signs
    as returned by two_coordinate_cycle, or any other valid alternating
    cycle). Raises if any resulting weight would go negative -- callers
    (e.g. random_two_coordinate_move) must pick a feasible epsilon."""
    combined = {}
    for point, weight in zip(map(tuple, mu.points), mu.weights):
        combined[point] = combined.get(point, 0.0) + weight
    for point, sign in zip(map(tuple, points), signs):
        combined[point] = combined.get(point, 0.0) + epsilon * sign

    out_points = np.array(list(combined.keys()), dtype=np.int64)
    out_weights = np.array(list(combined.values()), dtype=np.float64)
    if np.any(out_weights < -_TOL):
        raise ValueError("cycle_move: epsilon too large, would go negative")
    keep = out_weights > _TOL
    return Measure(points=out_points[keep], weights=out_weights[keep], n=mu.n)


def random_two_coordinate_move(mu, rng, epsilon_fraction=0.5, max_attempts=20):
    """Convenience local move for search/annealing: find two existing,
    positive-weight support points that already agree on 4 of the 6
    coordinates (i.e. differ only in some pair (a, b)), and shift
    epsilon = epsilon_fraction * min(their two weights) around the
    2-coordinate cycle through them (see two_coordinate_cycle). Returns
    (new_measure, epsilon) or None if no such pair is found within
    `max_attempts` random tries -- known limitation: for a generic support
    with no two points sharing 4 coordinates, this move is inapplicable (see
    module docstring on cycle scope); callers should retry or fall back."""
    m = mu.support_size
    if m < 2:
        return None
    for _ in range(max_attempts):
        coord_a, coord_b = rng.choice(N_COORDS, size=2, replace=False)
        rest_coords = [c for c in range(N_COORDS) if c not in (coord_a, coord_b)]
        rest_projection = mu.points[:, rest_coords]
        i = rng.integers(0, m)
        matches = np.flatnonzero(np.all(rest_projection == rest_projection[i], axis=1))
        matches = matches[matches != i]
        if matches.size == 0:
            continue
        j = matches[rng.integers(0, matches.size)]
        a_i, b_i = mu.points[i, coord_a], mu.points[i, coord_b]
        a_j, b_j = mu.points[j, coord_a], mu.points[j, coord_b]
        if a_i == a_j or b_i == b_j:
            continue  # points coincide or already share one of the two axes

        points, signs = two_coordinate_cycle(
            rest_coords, mu.points[i, rest_coords], coord_a, coord_b, [a_i, a_j], [b_i, b_j]
        )
        epsilon = epsilon_fraction * min(mu.weights[i], mu.weights[j])
        if epsilon <= _TOL:
            continue
        # i and j themselves sit at the "+1" cycle positions (see
        # two_coordinate_cycle); negate epsilon so they are the ones losing
        # weight into the two (until now empty) cross corners.
        return cycle_move(mu, points, signs, -epsilon), epsilon
    return None


def pushforward(mu, coordinate_perms):
    """Apply a per-coordinate bijection of [0, n) to mu's support: the
    pushforward mu = f_* nu for f = product of the given 1-D bijections.
    `coordinate_perms` is a length-6 sequence where entry c is either None
    (identity on that axis) or a length-n int array/list mapping grid index
    -> grid index. Any bijection of [0, n) preserves the uniform marginal on
    that axis, so the result stays in A_n. Generalizes
    fw.symmetry.apply_transform (axis permutation / global reflection) to
    arbitrary per-axis grid bijections, e.g. for warping/symmetry
    experiments."""
    if len(coordinate_perms) != N_COORDS:
        raise ValueError(f"coordinate_perms must have length {N_COORDS}")
    points = mu.points.copy()
    for coord, perm in enumerate(coordinate_perms):
        if perm is None:
            continue
        perm = np.asarray(perm, dtype=np.int64)
        if sorted(perm.tolist()) != list(range(mu.n)):
            raise ValueError(f"coordinate_perms[{coord}] must be a bijection of range(n)")
        points[:, coord] = perm[points[:, coord]]
    return Measure(points=points, weights=mu.weights.copy(), n=mu.n)
