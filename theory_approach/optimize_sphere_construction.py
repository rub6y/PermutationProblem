"""d=3 test of the growing-gadget hypothesis (01_growing_gadget_amplification.md
S:6): n points on the unit sphere S^2 (standing in for a generic convex point
set / point measure in R^3), 6 fixed unit direction vectors in R^3 (not
constrained to a single rotating plane, unlike d=2's circle-of-directions).
Sorting by dot product with each direction gives 6 permutations of [n]; J is
evaluated exactly with the project's fw.shattering predicate, and both the
point positions and the 6 directions are optimized jointly by simulated
annealing, exactly mirroring optimize_circular_construction.py's approach so
the d=2 (J~0.2498) and d=3 results are directly comparable.
"""

import sys
import pathlib
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fw.shattering import count_shattered_triples  # noqa: E402


def fibonacci_sphere(n):
    i = np.arange(n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    golden = np.pi * (3 - np.sqrt(5))
    theta = golden * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


def ranks_from_vectors(pts, dirs):
    """pts: (n,3) unit vectors. dirs: (n_dirs,3) unit vectors. -> (n,n_dirs) int64 ranks."""
    n = pts.shape[0]
    n_dirs = dirs.shape[0]
    proj = pts @ dirs.T  # (n, n_dirs)
    ranks = np.empty((n, n_dirs), dtype=np.int64)
    for k in range(n_dirs):
        ranks[:, k] = np.argsort(np.argsort(proj[:, k]))
    return ranks


def J_of(pts, dirs):
    n = pts.shape[0]
    points = ranks_from_vectors(pts, dirs)
    return count_shattered_triples(points) / n**3


def normalize_rows(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def anneal(n=60, n_dirs=6, steps=150000, seed=0, T0=0.006, T1=1e-5,
           sigma=0.3, verbose_every=None):
    rng = np.random.default_rng(seed)
    pts = fibonacci_sphere(n)
    # octahedron directions as the natural "evenly spaced" d=3 baseline
    dirs = normalize_rows(np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    ], dtype=np.float64))

    J_cur = J_of(pts, dirs)
    baseline_J = J_cur
    best_pts, best_dirs, best_J = pts.copy(), dirs.copy(), J_cur

    t0 = time.time()
    for step in range(steps):
        T = T0 * (T1 / T0) ** (step / steps)
        move_pt = rng.random() < 0.85
        if move_pt:
            i = rng.integers(n)
            old_val = pts[i].copy()
            pts[i] = normalize_rows(pts[i] + rng.normal(0, sigma, size=3))
        else:
            i = rng.integers(n_dirs)
            old_val = dirs[i].copy()
            dirs[i] = normalize_rows(dirs[i] + rng.normal(0, sigma, size=3))

        J_new = J_of(pts, dirs)
        d = J_new - J_cur
        if d >= 0 or rng.random() < np.exp(d / T):
            J_cur = J_new
            if J_cur > best_J:
                best_J, best_pts, best_dirs = J_cur, pts.copy(), dirs.copy()
        else:
            if move_pt:
                pts[i] = old_val
            else:
                dirs[i] = old_val

        if verbose_every and (step + 1) % verbose_every == 0:
            print(f"step {step+1:6d} T={T:.5f} J_cur={J_cur:.5f} best={best_J:.5f} "
                  f"({time.time()-t0:.1f}s)", flush=True)

    return best_pts, best_dirs, best_J, baseline_J


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 150000
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    best_pts, best_dirs, best_J, baseline_J = anneal(
        n=n, steps=steps, seed=seed, verbose_every=max(1, steps // 10)
    )
    print(f"n={n} baseline(octahedron dirs, fibonacci sphere) J={baseline_J:.5f}")
    print(f"n={n} seed={seed} annealed best J={best_J:.6f}")
