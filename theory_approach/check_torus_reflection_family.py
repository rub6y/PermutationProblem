"""Numerical check for 03_torus_reflection_pairs.md.

Family: mu_beta is the pushforward of Uniform[0,1) under
  t -> ( t, 1-t, frac(b2*t), 1-frac(b2*t), frac(b3*t), 1-frac(b3*t) )
for real slopes b2, b3 (b1 fixed to 1 by rescaling time). This is the
continuum limit of the "per-pair affine map plus reversal" algebraic family
already tried at small n and reported as a dead end
(records_best_structure.md, algebraic_construction_search.py).

Claim proved in the theory note: because coordinates 2k-1,2k are always
exact order-reversals of each other, shattering a triple reduces to asking
that the THREE order types (of t, of frac(b2 t), of frac(b3 t)) fall into
three different classes of the order_code reversal-pairing {0,5},{1,4},{2,3}
-- i.e. this whole 6-dimensional problem collapses to a 3-variable
equidistribution question. This script estimates J(b2,b3) by Monte Carlo
over a grid of slopes to see whether this exactly-parametrized family can
approach known records/0.5, or plateaus like every other closed form tried.
"""

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fw.shattering import order_code  # numba njit function, callable from Python


def J_of_beta(b2, b3, n_samples=400_000, seed=0):
    rng = np.random.default_rng(seed)
    tx = rng.uniform(0, 1, n_samples)
    ty = rng.uniform(0, 1, n_samples)
    tz = rng.uniform(0, 1, n_samples)

    def classes(scale):
        a = (scale * tx) % 1.0
        b = (scale * ty) % 1.0
        c = (scale * tz) % 1.0
        codes = np.empty(n_samples, dtype=np.int64)
        for i in range(n_samples):
            codes[i] = order_code(a[i], b[i], c[i])
        return np.minimum(codes, 5 - codes)

    c1 = classes(1.0)
    c2 = classes(b2)
    c3 = classes(b3)
    hits = (c1 != c2) & (c1 != c3) & (c2 != c3)
    return hits.mean()


if __name__ == "__main__":
    # Small integer slopes first (matches the discrete family already tried),
    # then irrational slopes (genuinely new region of the family).
    candidates = [
        (2.0, 3.0), (2.0, 4.0), (3.0, 5.0), (3.0, 7.0), (5.0, 7.0),
        (2.0, 1.6180339887), (3.0, 1.6180339887 * 2),
        (1.6180339887, 2.2360679775),  # golden ratio, sqrt(5)
        (2.4142135624, 3.3027756377),  # 1+sqrt2, sqrt(2)+sqrt(5)... exploratory
    ]
    best = (None, -1.0)
    for b2, b3 in candidates:
        j = J_of_beta(b2, b3, n_samples=200_000)
        print(f"b2={b2:.6f} b3={b3:.6f}  J~{j:.5f}")
        if j > best[1]:
            best = ((b2, b3), j)
    print("best so far:", best)
