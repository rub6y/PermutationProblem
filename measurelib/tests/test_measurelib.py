"""Cross-checks measurelib against the existing, working fw/ implementation
(same witness data, independently implemented shattering predicate and J),
plus regression tests for the new shift/hyperplane/features primitives.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fw.measure import load_permutations_as_measure as fw_load_witness
from fw.objective import compute_J as fw_compute_J

from measurelib.measure import Measure, check_marginals, load_witness, marginal_masses
from measurelib.functional import J, shattered
from measurelib.shift import (
    cycle_move,
    interpolate,
    pushforward,
    random_two_coordinate_move,
    two_coordinate_cycle,
)
from measurelib.features import hyperplane_residual, pca
from measurelib.hyperplane import integrate_subset, restrict_to_hyperplane, slice_integral

WITNESS_PATH = REPO_ROOT.parent / "good_permutation.txt"


def test_J_matches_fw_on_witness():
    mu = load_witness(WITNESS_PATH)
    fw_mu = fw_load_witness(WITNESS_PATH)
    assert abs(J(mu) - fw_compute_J(fw_mu)) < 1e-12


def test_shattered_matches_fw_on_random_triples():
    from fw.shattering import is_shattered_triple

    rng = np.random.default_rng(0)
    mismatches = 0
    for _ in range(2000):
        x1, x2, x3 = rng.integers(0, 30, size=(3, 6))
        if shattered(x1, x2, x3) != is_shattered_triple(x1, x2, x3):
            mismatches += 1
    assert mismatches == 0


def test_witness_is_in_A_n():
    mu = load_witness(WITNESS_PATH)
    check_marginals(mu)  # raises on failure


def test_two_coordinate_cycle_preserves_marginals():
    # Synthetic measure with two points (rows 0, 1) that agree on all
    # coordinates except {0, 1} -- the structural precondition
    # random_two_coordinate_move needs -- so the move is guaranteed to
    # apply, not left to chance on a generic witness's support.
    points = np.array(
        [
            [0, 0, 5, 5, 5, 5],
            [1, 1, 5, 5, 5, 5],
            [2, 2, 2, 2, 2, 2],
        ],
        dtype=np.int64,
    )
    weights = np.array([0.4, 0.4, 0.2])
    mu = Measure(points=points, weights=weights, n=6)
    before = marginal_masses(mu)

    rng = np.random.default_rng(1)
    n_applied = 0
    for _ in range(50):
        result = random_two_coordinate_move(mu, rng, epsilon_fraction=0.3)
        if result is None:
            continue
        moved, epsilon = result
        assert epsilon > 0
        assert np.max(np.abs(marginal_masses(moved) - before)) < 1e-9
        assert np.all(moved.weights >= -1e-12)
        n_applied += 1
    assert n_applied > 0  # the move must actually apply at least once


def test_cycle_move_rejects_infeasible_epsilon():
    mu = load_witness(WITNESS_PATH)
    rest_coords = [2, 3, 4, 5]
    points, signs = two_coordinate_cycle(
        rest_coords, mu.points[0, rest_coords], 0, 1, [0, 1], [0, 1]
    )
    try:
        cycle_move(mu, points, signs, epsilon=10.0)
        assert False, "expected ValueError for infeasible epsilon"
    except ValueError:
        pass


def test_interpolate_endpoints():
    mu = load_witness(WITNESS_PATH)
    at_zero = interpolate(mu, mu, 0.0)
    assert abs(J(at_zero) - J(mu)) < 1e-12


def test_pushforward_identity():
    mu = load_witness(WITNESS_PATH)
    identity = [None] * 6
    same = pushforward(mu, identity)
    assert np.array_equal(np.sort(same.points, axis=0), np.sort(mu.points, axis=0))


def test_hyperplane_residual_near_zero_for_known_witness():
    # KNOWLEDGE.md SS4: good_permutation.txt is a near-optimal witness known
    # to lie exactly on the central hyperplane sum_k x_k = 3(n-1).
    mu = load_witness(WITNESS_PATH)
    residual = hyperplane_residual(mu)
    assert np.max(np.abs(residual)) < 1e-9


def test_restrict_to_hyperplane_keeps_full_support_for_witness():
    mu = load_witness(WITNESS_PATH)
    on_h = restrict_to_hyperplane(mu)
    assert on_h.support_size == mu.support_size


def test_slice_and_subset_integrals_agree():
    mu = load_witness(WITNESS_PATH)
    total = sum(slice_integral(mu, 0, v) for v in range(mu.n))
    assert abs(total - mu.weights.sum()) < 1e-12
    subset_mass = integrate_subset(mu, lambda points: points[:, 0] < mu.n // 2)
    assert 0.0 < subset_mass < mu.weights.sum()


def test_pca_eigenvalues_sorted_ascending():
    mu = load_witness(WITNESS_PATH)
    eigenvalues, eigenvectors = pca(mu)
    assert np.all(np.diff(eigenvalues) >= -1e-9)
    assert eigenvectors.shape == (6, 6)


if __name__ == "__main__":
    # no pytest in shell.nix; run directly, matching tests/test_shattering.py
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: passed")
    print("test_measurelib.py: all tests passed")
