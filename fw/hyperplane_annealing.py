"""Simulated annealing on fw.hyperplane_search's move class (2-point double
transpositions that preserve every row's sum, i.e. stay exactly on the
central hyperplane sum_k x_k(i) = 3(n-1)).

fw.hyperplane_search.hill_climb_hyperplane is pure greedy: it takes 0 steps
from good_permutation.txt (no single double-swap improves it -- confirmed in
FWPush.md/symmetry_search.md). This module adds Metropolis acceptance so the
walk can cross shallow dips instead of stopping at the first local optimum.
"""

from dataclasses import dataclass, field

import numpy as np

from fw.hyperplane_search import (
    _candidate_k_cycles,
    _candidate_moves,
    _evaluate_kcycle_moves,
    _evaluate_moves,
    apply_double_swap,
    apply_kcycle,
)
from fw.shattering import count_shattered_triples


@dataclass
class AnnealResult:
    best_points: np.ndarray
    best_J: float
    history: list = field(default_factory=list)  # J at every accepted/rejected step


def anneal_hyperplane(points, n_steps=3000, t_start=8.0, t_end=0.05, seed=None):
    """Metropolis annealing over hyperplane-preserving double-swaps.
    Temperature cools geometrically from t_start to t_end over n_steps.
    delta is in units of shattered-triple *count* (integer), matching the
    scale _evaluate_moves already works in, so T is on that same scale."""
    rng = np.random.default_rng(seed)
    points = points.copy()
    n = points.shape[0]

    current_count = count_shattered_triples(points)
    best_points, best_count = points.copy(), current_count
    history = [current_count / n**3]

    cooling = (t_end / t_start) ** (1.0 / max(n_steps - 1, 1))
    t = t_start

    for step in range(n_steps):
        moves = _candidate_moves(points)
        if not moves:
            break
        ks = np.array([m[0] for m in moves], dtype=np.int64)
        ls = np.array([m[1] for m in moves], dtype=np.int64)
        is_ = np.array([m[2] for m in moves], dtype=np.int64)
        js_ = np.array([m[3] for m in moves], dtype=np.int64)
        deltas = _evaluate_moves(points, ks, ls, is_, js_)

        # Metropolis: prefer improving moves, but allow worsening ones with
        # probability exp(delta / t). Sample proportional to acceptance
        # weight among all candidates rather than always taking the argmax,
        # so the walk can explore rather than immediately re-greedifying.
        weights = np.exp(np.clip(deltas / t, -50, 50))
        weights = weights / weights.sum()
        idx = rng.choice(len(moves), p=weights)
        delta = int(deltas[idx])

        accept = delta >= 0 or rng.random() < np.exp(delta / t)
        if accept:
            k, l, i, j = ks[idx], ls[idx], is_[idx], js_[idx]
            points = apply_double_swap(points, int(k), int(l), int(i), int(j))
            current_count += delta
            if current_count > best_count:
                best_count = current_count
                best_points = points.copy()

        history.append(current_count / n**3)
        t *= cooling

    return AnnealResult(best_points=best_points, best_J=best_count / n**3, history=history)


def anneal_hyperplane_multimove(points, n_steps=3000, t_start=8.0, t_end=0.05,
                                 k_sizes=(2, 3, 4), max_per_group=200, seed=None):
    """Like anneal_hyperplane, but the candidate pool at each step is the
    union of 2-point double-swaps and k-cycles for k in k_sizes (Track 3):
    larger hyperplane-preserving moves reach configurations no sequence of
    2-point moves can reach in a single step."""
    rng = np.random.default_rng(seed)
    points = points.copy()
    n = points.shape[0]

    current_count = count_shattered_triples(points)
    best_points, best_count = points.copy(), current_count
    history = [current_count / n**3]

    cooling = (t_end / t_start) ** (1.0 / max(n_steps - 1, 1))
    t = t_start

    for step in range(n_steps):
        # kind 0: double-swap (k, l, i, j); kind >0: k_size-cycle (k, l, idxs)
        kinds, ks, ls, applys, deltas_list = [], [], [], [], []

        moves2 = _candidate_moves(points)
        if moves2:
            ks2 = np.array([m[0] for m in moves2], dtype=np.int64)
            ls2 = np.array([m[1] for m in moves2], dtype=np.int64)
            is_ = np.array([m[2] for m in moves2], dtype=np.int64)
            js_ = np.array([m[3] for m in moves2], dtype=np.int64)
            deltas2 = _evaluate_moves(points, ks2, ls2, is_, js_)
            for idx in range(len(moves2)):
                kinds.append(0)
                ks.append(ks2[idx]); ls.append(ls2[idx])
                applys.append((int(is_[idx]), int(js_[idx])))
                deltas_list.append(deltas2[idx])

        for k_size in k_sizes:
            if k_size == 2:
                continue
            moves_k = _candidate_k_cycles(points, k_size, max_per_group=max_per_group, rng=rng)
            if not moves_k:
                continue
            ksk = np.array([m[0] for m in moves_k], dtype=np.int64)
            lsk = np.array([m[1] for m in moves_k], dtype=np.int64)
            idxs_matrix = np.array([m[2] for m in moves_k], dtype=np.int64)
            deltask = _evaluate_kcycle_moves(points, ksk, lsk, idxs_matrix)
            for idx in range(len(moves_k)):
                kinds.append(k_size)
                ks.append(ksk[idx]); ls.append(lsk[idx])
                applys.append(idxs_matrix[idx])
                deltas_list.append(deltask[idx])

        if not deltas_list:
            break

        deltas = np.array(deltas_list, dtype=np.int64)
        weights = np.exp(np.clip(deltas / t, -50, 50))
        weights = weights / weights.sum()
        idx = rng.choice(len(deltas), p=weights)
        delta = int(deltas[idx])

        accept = delta >= 0 or rng.random() < np.exp(delta / t)
        if accept:
            kind = kinds[idx]
            if kind == 0:
                i, j = applys[idx]
                points = apply_double_swap(points, int(ks[idx]), int(ls[idx]), i, j)
            else:
                points = apply_kcycle(points, int(ks[idx]), int(ls[idx]), applys[idx])
            current_count += delta
            if current_count > best_count:
                best_count = current_count
                best_points = points.copy()

        history.append(current_count / n**3)
        t *= cooling

    return AnnealResult(best_points=best_points, best_J=best_count / n**3, history=history)


def multi_start_anneal_multimove(points, n_runs=5, n_steps=3000, seed=None, **kwargs):
    rng = np.random.default_rng(seed)
    best_result = None
    for _ in range(n_runs):
        result = anneal_hyperplane_multimove(points, n_steps=n_steps,
                                              seed=int(rng.integers(0, 1 << 30)), **kwargs)
        if best_result is None or result.best_J > best_result.best_J:
            best_result = result
    return best_result


def multi_start_anneal(points, n_runs=5, n_steps=3000, seed=None, **kwargs):
    """Several independent annealing runs from the same seed point, keeping
    the best result across runs (SA's per-run variance is high, so multiple
    independent runs matter more than a single very long one)."""
    rng = np.random.default_rng(seed)
    best_result = None
    for _ in range(n_runs):
        result = anneal_hyperplane(points, n_steps=n_steps, seed=int(rng.integers(0, 1 << 30)), **kwargs)
        if best_result is None or result.best_J > best_result.best_J:
            best_result = result
    return best_result
