"""Falsification check for 01_growing_gadget_amplification.md.

Two independent computations of J for the d=2 "convex position + 6 rotating
directions" construction (uniform point measure, 6 evenly spaced directions):

1. Discrete: build n points on a circle (golden-angle spacing, for general
   position), sort by 6 evenly spaced directions to get 6 permutations of
   [n], and evaluate J with the project's own fw.shattering predicate.
2. Continuous (Monte Carlo): sample random angle triples directly and apply
   the midpoint/arc "rainbow" criterion derived in the theory note.

If these disagree beyond Monte Carlo/discretization noise, the geometric
formula in the theory note has a bug and must be fixed before trusting any
further conclusion drawn from it.
"""

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fw.shattering import count_shattered_triples  # noqa: E402

GOLDEN_ANGLE = np.pi * (3 - np.sqrt(5))


def discrete_J(n, n_dirs=6):
    angles = (np.arange(n) * GOLDEN_ANGLE) % (2 * np.pi)
    dirs = np.arange(n_dirs) * (2 * np.pi / n_dirs)
    points = np.zeros((n, n_dirs), dtype=np.int64)
    for k, theta in enumerate(dirs):
        proj = np.cos(theta) * np.cos(angles) + np.sin(theta) * np.sin(angles)
        points[:, k] = np.argsort(np.argsort(proj))
    count = count_shattered_triples(points)
    return count / n**3


def rainbow(phi_x, phi_y, phi_z, dirs):
    m = np.array(
        [
            (phi_x + phi_y) / 2 % np.pi,
            (phi_x + phi_z) / 2 % np.pi,
            (phi_y + phi_z) / 2 % np.pi,
        ]
    )
    crit = np.sort(np.concatenate([m, m + np.pi]) % (2 * np.pi))
    # arc index of each fixed direction: how many critical points precede it (mod 6)
    idx = np.searchsorted(crit, dirs, side="right") % 6
    return len(set(idx.tolist())) == 6


def continuous_J(n_samples, n_dirs=6, seed=0):
    rng = np.random.default_rng(seed)
    dirs = np.arange(n_dirs) * (2 * np.pi / n_dirs)
    hits = 0
    for _ in range(n_samples):
        phi = rng.uniform(0, 2 * np.pi, size=3)
        if rainbow(phi[0], phi[1], phi[2], dirs):
            hits += 1
    return hits / n_samples


if __name__ == "__main__":
    for n in (30, 60, 120, 240):
        print(f"discrete   n={n:4d}: J = {discrete_J(n):.6f}")
    print(f"continuous MC (2,000,000 samples): J = {continuous_J(2_000_000):.6f}")
