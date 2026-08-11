"""Optimize the free parameters of the d=2 circular construction
(01_growing_gadget_amplification.md, step 5.2): the point-angle measure rho
(here: the actual angles of n points on the circle, standing in for its
finite-n discretization) and the 6 sort directions Theta.

The naive symmetric baseline (golden-angle points, Theta evenly spaced by
60 deg) gives J0 = 2/9 ~ 0.2222 (checked in check_circular_construction.py).
This script runs simulated annealing directly on the discrete objective
(fw.shattering.count_shattered_triples) over both the n point-angles and
the 6 direction-angles jointly, to see how much of the gap to the record
(0.495195) and to 0.5 can be closed by optimizing within this family before
concluding anything about needing d>2.
"""

import sys
import pathlib
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fw.shattering import count_shattered_triples  # noqa: E402

GOLDEN_ANGLE = np.pi * (3 - np.sqrt(5))


def ranks_from_angles(phi, theta):
    """phi: (n,) point angles. theta: (n_dirs,) direction angles.
    Returns (n, n_dirs) int64 rank matrix."""
    n = phi.shape[0]
    n_dirs = theta.shape[0]
    points = np.empty((n, n_dirs), dtype=np.int64)
    cphi, sphi = np.cos(phi), np.sin(phi)
    for k in range(n_dirs):
        proj = np.cos(theta[k]) * cphi + np.sin(theta[k]) * sphi
        points[:, k] = np.argsort(np.argsort(proj))
    return points


def J_of(phi, theta):
    points = ranks_from_angles(phi, theta)
    n = phi.shape[0]
    return count_shattered_triples(points) / n**3


def anneal(n=60, n_dirs=6, steps=20000, seed=0, T0=0.02, T1=1e-4, verbose_every=2000):
    rng = np.random.default_rng(seed)
    phi = (np.arange(n) * GOLDEN_ANGLE) % (2 * np.pi)
    theta = np.arange(n_dirs) * (2 * np.pi / n_dirs)

    J_cur = J_of(phi, theta)
    best_phi, best_theta, best_J = phi.copy(), theta.copy(), J_cur

    t0 = time.time()
    for step in range(steps):
        T = T0 * (T1 / T0) ** (step / steps)
        # perturb either a point angle or a direction angle
        move_phi = rng.random() < 0.85
        if move_phi:
            i = rng.integers(n)
            old_val = phi[i]
            phi[i] = (phi[i] + rng.normal(0, 0.3)) % (2 * np.pi)
        else:
            i = rng.integers(n_dirs)
            old_val = theta[i]
            theta[i] = (theta[i] + rng.normal(0, 0.3)) % (2 * np.pi)

        J_new = J_of(phi, theta)
        d = J_new - J_cur
        if d >= 0 or rng.random() < np.exp(d / T):
            J_cur = J_new
            if J_cur > best_J:
                best_J, best_phi, best_theta = J_cur, phi.copy(), theta.copy()
        else:
            if move_phi:
                phi[i] = old_val
            else:
                theta[i] = old_val

        if (step + 1) % verbose_every == 0:
            print(f"step {step+1:6d}  T={T:.5f}  J_cur={J_cur:.5f}  best={best_J:.5f}  "
                  f"({time.time()-t0:.1f}s)")

    return best_phi, best_theta, best_J


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    print(f"baseline (golden-angle, evenly-spaced dirs) at n={n}: "
          f"J = {J_of((np.arange(n)*GOLDEN_ANGLE) % (2*np.pi), np.arange(6)*(2*np.pi/6)):.5f}")
    best_phi, best_theta, best_J = anneal(n=n, steps=steps)
    print(f"annealed best J = {best_J:.6f}")
    print("theta (deg):", np.sort(np.degrees(best_theta) % 360))
