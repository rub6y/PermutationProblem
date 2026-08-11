"""Regression check: fw.shattering.is_shattered_triple must agree with the
independent pure-Python reference implementation on random triples."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fw.shattering import is_shattered_triple, shattering_reference


def test_matches_reference_on_random_triples():
    rng = np.random.default_rng(0)
    mismatches = 0
    for _ in range(2000):
        x1, x2, x3 = rng.integers(0, 30, size=(3, 6))
        fast = is_shattered_triple(x1, x2, x3)
        slow = shattering_reference(x1, x2, x3)
        if fast != slow:
            mismatches += 1
    assert mismatches == 0


def test_matches_reference_on_shattered_construction():
    # deliberately construct a triple that is shattered: for each of the 6
    # coordinates, assign values 0/1/2 to x1/x2/x3 according to a distinct
    # one of the 6 total orders on {x1, x2, x3}
    orders = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    points = np.zeros((3, 6), dtype=np.int64)
    for coord, order in enumerate(orders):
        for rank, row in enumerate(order):
            points[row, coord] = rank
    x1, x2, x3 = points[0], points[1], points[2]
    assert is_shattered_triple(x1, x2, x3) == shattering_reference(x1, x2, x3) == True


if __name__ == "__main__":
    # no pytest in shell.nix; run directly so `python3 tests/test_shattering.py` works
    test_matches_reference_on_random_triples()
    test_matches_reference_on_shattered_construction()
    print("test_shattering.py: all tests passed")
