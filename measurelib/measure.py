"""Representation of A_n: finitely supported approximations, by grid squares,
of Borel probability measures on [0,1]^6 with all 6 marginals uniform.

Grid index i in [0, n) along an axis stands for the sub-square [i/n, (i+1)/n)
of that axis; a Measure is a sparse (point, weight) list over such squares
rather than a dense n^6 array, since n grows into the thousands in practice
(a dense array is infeasible past n ~ 10).

This module is independent of fw/measure.py by design (see measurelib's
design note): fw/ stays frozen as the working search pipeline that produced
the current best record, and measurelib is the general-purpose foundation
future work (search, ML, construction, proof) builds on.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_COORDS = 6


@dataclass
class Measure:
    """A finitely supported measure on [0, n)^6.

    points: int64 array of shape (M, 6), grid cell indices in [0, n).
    weights: float64 array of shape (M,), nonnegative.
    n: grid resolution.
    """

    points: np.ndarray
    weights: np.ndarray
    n: int

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.int64)
        self.weights = np.asarray(self.weights, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != N_COORDS:
            raise ValueError(f"points must have shape (M, {N_COORDS})")
        if self.weights.shape != (self.points.shape[0],):
            raise ValueError("weights must have shape (M,) matching points")

    @property
    def support_size(self):
        return self.points.shape[0]


def from_permutations(permutations):
    """Six permutations of [n] (one per coordinate) -> the uniform measure on
    their n grid points (the discretized central-hyperplane-style witnesses
    used throughout this project, e.g. a_summary.txt / good_permutation.txt)."""
    n = len(permutations[0])
    points = np.array(permutations, dtype=np.int64).T  # shape (n, 6)
    weights = np.full(n, 1.0 / n, dtype=np.float64)
    return Measure(points=points, weights=weights, n=n)


def load_witness(path):
    """Parse a `S1 = [(perm), ...]` witness file (a_summary.txt /
    good_permutation.txt format) into the uniform measure on its n points."""
    text = Path(path).read_text()
    namespace = {}
    exec(compile(text, str(path), "exec"), namespace)
    permutations = next(v for k, v in namespace.items() if not k.startswith("__"))
    if len(permutations) != N_COORDS:
        raise ValueError(f"expected {N_COORDS} permutations, got {len(permutations)}")
    n = len(permutations[0])
    for perm in permutations:
        if sorted(perm) != list(range(n)):
            raise ValueError("each row must be a permutation of range(n)")
    return from_permutations(permutations)


_A_SUMMARY_HEADER = re.compile(r"^n\s*=\s*(\d+).*\(([\d.]+)\)\s*$")


def load_a_summary_series(path):
    """Parse a_summary.txt's format: repeated blocks of a `n = <n> ...
    (<J>)` header (the "-->" between the count and the parenthesized J is
    present in most but not all blocks, so the regex ignores it) followed by
    N_COORDS whitespace-separated-integer permutation rows. Returns
    {n: (Measure, reported_J)} for every block found."""
    lines = Path(path).read_text().splitlines()
    result = {}
    i = 0
    while i < len(lines):
        header = _A_SUMMARY_HEADER.match(lines[i])
        if header:
            n = int(header.group(1))
            reported_J = float(header.group(2))
            perms = []
            for _ in range(N_COORDS):
                i += 1
                perms.append(tuple(int(x) for x in lines[i].split()))
            # Most blocks are 0-indexed (range(n)); at least one (n=26) is
            # 1-indexed (range(1, n+1)) -- normalize by each row's own min
            # rather than trust a fixed convention.
            perms = [tuple(v - min(row) for v in row) for row in perms]
            result[n] = (from_permutations(perms), reported_J)
        i += 1
    return result


def load_records_jsonl(path):
    """Parse records.jsonl: each line is a JSON record with a `config`
    field, an (M, 6) array. The log is inconsistent about units: most
    entries store raw grid indices directly (integers in [0, M)), but at
    least one (the original S1-seeded record) stores normalized floats in
    [0, 1] on step 1/(M-1) instead -- detected here by whether the max value
    is <= 1.0 (raw-index configs go up to M-1 >= 1 for any M > 2). Yields
    (Measure, record) for every line that has a `config` field."""
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        config = record.get("config")
        if config is None:
            continue
        m = record["M"]
        arr = np.array(config, dtype=np.float64)
        if arr.max() <= 1.0:
            arr = arr * (m - 1)
        points = np.rint(arr).astype(np.int64)
        weights = np.full(m, 1.0 / m, dtype=np.float64)
        yield Measure(points=points, weights=weights, n=m), record


def marginal_masses(measure):
    """For each of the 6 coordinates, the total weight landing in each of the
    n grid cells (should be uniform 1/n for a measure in A_n)."""
    n = measure.n
    masses = np.zeros((N_COORDS, n), dtype=np.float64)
    for coord in range(N_COORDS):
        np.add.at(masses[coord], measure.points[:, coord], measure.weights)
    return masses


def check_marginals(measure, tol=1e-9):
    """Raise if `measure` is not (approximately) in A_n: nonnegative weights
    summing to 1 with all 6 marginals uniform. Returns the max marginal error
    on success, so callers can log/assert on it."""
    if np.any(measure.weights < -tol):
        raise ValueError("weights must be nonnegative")
    total = measure.weights.sum()
    if abs(total - 1.0) > tol:
        raise ValueError(f"weights must sum to 1, got {total}")
    target = 1.0 / measure.n
    masses = marginal_masses(measure)
    max_err = np.max(np.abs(masses - target))
    if max_err > tol:
        raise ValueError(f"marginal {max_err} away from uniform 1/n (tol={tol})")
    return max_err
