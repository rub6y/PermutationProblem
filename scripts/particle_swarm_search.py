"""Particle-based search experiments, compared against the existing
`fw.hyperplane_search.basin_hop_hyperplane` baseline.

Context: the feasible set (6-tuples of permutations of [n], optionally
restricted to the central hyperplane sum_k sigma_k(i) = 3(n-1)) is a finite,
disconnected lattice, not a vector space -- there is no well-defined
"continuous gradient step" a particle could take. So "particles moving with
simple steps" here always means a single discrete hyperplane-preserving
double-swap (fw.hyperplane_search's move set); the two approaches below only
differ in how they *choose* which swap to propose and how many independent
walkers they run:

  * `population_swarm`: many independent point-cloud candidates ("lots of
    pseudo-randomly scattered points") each doing small Metropolis steps in
    parallel, periodically culled (worst half replaced by mutated clones of
    the best half) -- a discrete stand-in for PSO's "move toward the best
    particle", since there's no velocity vector to interpolate between two
    permutations.
  * `gradient_biased_walk`: a single point cloud where fw.objective's exact
    functional derivative (the same one the Frank-Wolfe LP oracle uses) ranks
    the n points by how much they currently contribute to J, and swap
    proposals are biased toward relocating the lowest-contributing points.
    The gradient is only used to *rank* discrete moves, never to move a point
    continuously.

Usage: nix-shell shell.nix --run "python3 scripts/particle_swarm_search.py"
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from fw.hyperplane_search import (
    _best_hyperplane_swap,
    _candidate_moves,
    _double_swap_delta,
    apply_double_swap,
    basin_hop_hyperplane,
)
from fw.measure import load_permutations, measure_from_permutations
from fw.objective import compute_gradient
from fw.shattering import count_shattered_triples
from fw.swap_search import exact_J
from scripts.gap_survey import hyperplane_seed

ROOT = Path(__file__).resolve().parent.parent
GOOD_PERM_PATH = ROOT.parent / "good_permutation.txt"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"


def _temperature(step, n_steps, t_start, t_end, elapsed_s=None, max_time_s=None):
    """Geometric cooling from t_start to t_end. Keyed to wall-clock progress
    when max_time_s is given (so open-ended `n_steps=10**9` time-bounded runs
    still cool down), otherwise to step count."""
    if max_time_s is not None:
        frac = min(1.0, elapsed_s / max_time_s)
    else:
        frac = step / max(1, n_steps - 1)
    return t_start * (t_end / t_start) ** frac


def population_swarm(
    seed_points,
    n_particles=8,
    n_steps=400,
    t_start=1.0,
    t_end=0.02,
    cull_every=40,
    cull_frac=0.5,
    mutation_swaps=2,
    seed=None,
    max_time_s=None,
):
    """Many independent hyperplane-preserving Metropolis walkers, culled
    periodically. Returns (best_points, best_J, history) where history is a
    list of (step, elapsed_s, best_J, mean_J, spread_J). Stops at n_steps or
    max_time_s, whichever comes first (temperature schedule is always keyed
    to n_steps, so max_time_s just truncates a run early)."""
    rng = np.random.default_rng(seed)
    n = seed_points.shape[0]
    n3 = n**3
    t0 = time.time()

    particles = [seed_points.copy() for _ in range(n_particles)]
    counts = [int(count_shattered_triples(p)) for p in particles]
    best_idx = int(np.argmax(counts))
    best_points = particles[best_idx].copy()
    best_count = counts[best_idx]
    history = []

    for step in range(n_steps):
        elapsed = time.time() - t0
        if max_time_s is not None and elapsed > max_time_s:
            break
        t = _temperature(step, n_steps, t_start, t_end, elapsed, max_time_s)
        for p_idx in range(n_particles):
            points = particles[p_idx]
            moves = _candidate_moves(points)
            if not moves:
                continue
            k, l, i, j = moves[rng.integers(0, len(moves))]
            delta = _double_swap_delta(points, k, l, i, j)
            accept = delta >= 0 or rng.random() < np.exp(delta / (t * n))
            if accept:
                particles[p_idx] = apply_double_swap(points, k, l, i, j)
                counts[p_idx] += delta
                if counts[p_idx] > best_count:
                    best_count = counts[p_idx]
                    best_points = particles[p_idx].copy()

        if (step + 1) % cull_every == 0:
            order = list(np.argsort(counts)[::-1])
            n_survive = max(1, int(n_particles * (1 - cull_frac)))
            survivors, dead = order[:n_survive], order[n_survive:]
            for d_rank, dead_idx in enumerate(dead):
                parent = survivors[d_rank % len(survivors)]
                clone = particles[parent].copy()
                for _ in range(mutation_swaps):
                    mv = _candidate_moves(clone)
                    if not mv:
                        break
                    k, l, i, j = mv[rng.integers(0, len(mv))]
                    clone = apply_double_swap(clone, k, l, i, j)
                particles[dead_idx] = clone
                counts[dead_idx] = int(count_shattered_triples(clone))

        js = [c / n3 for c in counts]
        history.append((step, time.time() - t0, best_count / n3, float(np.mean(js)), float(np.max(js) - np.min(js))))

    return best_points, best_count / n3, history


def gradient_biased_walk(
    seed_points,
    n_steps=400,
    t_start=1.0,
    t_end=0.02,
    top_k_frac=0.25,
    regrad_every=10,
    seed=None,
    max_time_s=None,
):
    """Single point cloud; every `regrad_every` steps, rank points by the
    exact functional-derivative fw.objective.compute_gradient and restrict
    swap proposals to the lowest-contributing quarter of points. Returns
    (best_points, best_J, history) where history is (step, elapsed_s,
    current_J, best_J). Stops at n_steps or max_time_s, whichever first."""
    rng = np.random.default_rng(seed)
    points = seed_points.copy()
    n = points.shape[0]
    n3 = n**3
    top_k = max(4, int(n * top_k_frac))
    t0 = time.time()

    count = int(count_shattered_triples(points))
    best_points, best_count = points.copy(), count
    history = []
    low_idx = set(range(n))

    for step in range(n_steps):
        elapsed = time.time() - t0
        if max_time_s is not None and elapsed > max_time_s:
            break
        t = _temperature(step, n_steps, t_start, t_end, elapsed, max_time_s)

        if step % regrad_every == 0:
            measure = measure_from_permutations(points.T.tolist())
            grad = compute_gradient(measure, points)
            low_idx = set(np.argsort(grad)[:top_k].tolist())

        moves = _candidate_moves(points)
        biased = [mv for mv in moves if mv[2] in low_idx or mv[3] in low_idx]
        pool = biased if biased else moves
        if not pool:
            history.append((step, time.time() - t0, count / n3, best_count / n3))
            continue

        k, l, i, j = pool[rng.integers(0, len(pool))]
        delta = _double_swap_delta(points, k, l, i, j)
        accept = delta >= 0 or rng.random() < np.exp(delta / (t * n))
        if accept:
            points = apply_double_swap(points, k, l, i, j)
            count += delta
            if count > best_count:
                best_count, best_points = count, points.copy()

        history.append((step, time.time() - t0, count / n3, best_count / n3))

    return best_points, best_count / n3, history


def _hill_climb_hyperplane_timed(points, t0, max_time_s, max_steps=100000):
    """Like fw.hyperplane_search.hill_climb_hyperplane, but checks the
    wall-clock budget after every individual swap step, not just between
    restarts. A single hill-climb call at n=78 from a cold random seed takes
    ~100s uninterrupted (each step exhaustively scores every
    hyperplane-preserving move) -- with the un-timed version, that alone blew
    the basin_hop baseline's nominal 25s budget by ~4x in every trial of the
    first multi-seed run (see particle_swarm_results.md)."""
    points = points.copy()
    n = points.shape[0]
    steps = 0
    while steps < max_steps:
        if time.time() - t0 > max_time_s:
            break
        best = _best_hyperplane_swap(points)
        if best is None:
            break
        k, l, i, j, delta = best
        if delta <= 0:
            break
        points = apply_double_swap(points, k, l, i, j)
        steps += 1
    return points, count_shattered_triples(points) / n**3, steps


def basin_hop_hyperplane_timed(seed_points, max_time_s, perturb_swaps=3, seed=None, max_hill_climb_steps=100000):
    """Thin wrapper around fw.hyperplane_search.basin_hop_hyperplane that
    keeps adding restarts until max_time_s elapses, recording (elapsed_s,
    best_J) after every restart -- so it can be compared wall-clock-for-
    wall-clock against the particle methods instead of restart-for-restart.
    Each individual hill-climb call is itself time-boxed via
    _hill_climb_hyperplane_timed so a single expensive climb can't overshoot
    the budget."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    n = seed_points.shape[0]

    best_points, best_J, _ = _hill_climb_hyperplane_timed(seed_points, t0, max_time_s, max_hill_climb_steps)
    history = [(time.time() - t0, best_J)]

    while time.time() - t0 < max_time_s:
        perturbed = best_points.copy()
        applied, attempts = 0, 0
        while applied < perturb_swaps and attempts < perturb_swaps * 20:
            attempts += 1
            moves = _candidate_moves(perturbed)
            if not moves:
                break
            k, l, i, j = moves[rng.integers(0, len(moves))]
            perturbed = apply_double_swap(perturbed, k, l, i, j)
            applied += 1

        candidate, candidate_J, _ = _hill_climb_hyperplane_timed(perturbed, t0, max_time_s, max_hill_climb_steps)
        if candidate_J > best_J:
            best_points, best_J = candidate, candidate_J
        history.append((time.time() - t0, best_J))

    return best_points, best_J, history


def _warmup_jit():
    """Trigger Numba JIT compilation for every hot path used by the three
    search methods before any timing starts. Without this, whichever method
    happens to run first in a fresh process pays the one-time compilation
    cost out of its own time budget -- confirmed as the cause of the n=78
    run-to-run variance in the first round of experiments (a same-seed repeat
    in an otherwise-fresh process landed at J=0.270 instead of 0.330; see
    particle_swarm_results.md's Caveat section)."""
    dummy = hyperplane_seed(10, np.random.default_rng(0))
    count_shattered_triples(dummy)
    moves = _candidate_moves(dummy)
    if moves:
        k, l, i, j = moves[0]
        _double_swap_delta(dummy, k, l, i, j)
    measure = measure_from_permutations(dummy.T.tolist())
    compute_gradient(measure, dummy)
    from fw.hyperplane_search import hill_climb_hyperplane
    hill_climb_hyperplane(dummy, max_steps=5)


def _thin(history, max_points=300):
    """Downsample a history list to at most max_points entries (always
    keeping the first and last), so multi-trial JSON dumps of long
    gradient_biased_walk runs (tens of thousands of steps) stay a
    reasonable size. Plotting only needs the shape of the curve, not every
    step."""
    if len(history) <= max_points:
        return history
    stride = len(history) / max_points
    idxs = sorted({int(i * stride) for i in range(max_points)} | {len(history) - 1})
    return [history[i] for i in idxs]


def _load_seed(n, rng_seed, warm_start=False):
    if warm_start and n == 26 and GOOD_PERM_PATH.exists():
        perms = load_permutations(GOOD_PERM_PATH)
        return np.array(perms, dtype=np.int64).T
    return hyperplane_seed(n, np.random.default_rng(rng_seed))


def run_experiment(n, time_budget_s, label_suffix="", warm_start=False):
    print(f"\n=== n={n}{label_suffix} (time budget {time_budget_s}s/method) ===")
    seed_points = _load_seed(n, rng_seed=0, warm_start=warm_start)
    print(f"seed J = {exact_J(seed_points):.6f}")
    results = {}

    t0 = time.time()
    base_points, base_J, base_hist = basin_hop_hyperplane_timed(seed_points, max_time_s=time_budget_s, seed=1)
    results["baseline_basin_hop"] = {"J": base_J, "history": _thin(base_hist), "wall_s": time.time() - t0}
    print(f"baseline basin_hop_hyperplane: J={base_J:.6f} ({results['baseline_basin_hop']['wall_s']:.1f}s)")

    t0 = time.time()
    pop_points, pop_J, pop_hist = population_swarm(
        seed_points, n_steps=10**9, max_time_s=time_budget_s, seed=2
    )
    results["population_swarm"] = {"J": pop_J, "history": _thin(pop_hist), "wall_s": time.time() - t0}
    print(f"population_swarm:              J={pop_J:.6f} ({results['population_swarm']['wall_s']:.1f}s)")

    t0 = time.time()
    grad_points, grad_J, grad_hist = gradient_biased_walk(
        seed_points, n_steps=10**9, max_time_s=time_budget_s, seed=3
    )
    results["gradient_biased_walk"] = {"J": grad_J, "history": _thin(grad_hist), "wall_s": time.time() - t0}
    print(f"gradient_biased_walk:          J={grad_J:.6f} ({results['gradient_biased_walk']['wall_s']:.1f}s)")

    best_label = max(results, key=lambda k: results[k]["J"])
    print(f"best of the three: {best_label} (J={results[best_label]['J']:.6f})")

    return results


def run_experiment_multiseed(n, time_budget_s, n_trials, label_suffix="", warm_start=False, base_seed=100):
    """Like run_experiment, but repeats each method n_trials times with
    distinct seeds (offset per trial and per method, so trials aren't
    correlated across methods) and reports mean/std/min/max J -- needed
    because a single wall-clock-timed trial can't distinguish algorithm
    quality from run-to-run timing noise (see particle_swarm_results.md)."""
    print(f"\n=== n={n}{label_suffix} (time budget {time_budget_s}s/method, {n_trials} trials) ===")
    seed_points = _load_seed(n, rng_seed=0, warm_start=warm_start)
    print(f"seed J = {exact_J(seed_points):.6f}")

    method_runners = {
        "baseline_basin_hop": lambda sd: basin_hop_hyperplane_timed(seed_points, max_time_s=time_budget_s, seed=sd),
        "population_swarm": lambda sd: population_swarm(seed_points, n_steps=10**9, max_time_s=time_budget_s, seed=sd),
        "gradient_biased_walk": lambda sd: gradient_biased_walk(seed_points, n_steps=10**9, max_time_s=time_budget_s, seed=sd),
    }

    results = {name: {"J_values": [], "wall_s": [], "histories": []} for name in method_runners}

    for trial in range(n_trials):
        for m_idx, (name, runner) in enumerate(method_runners.items()):
            trial_seed = base_seed + trial * 10 + m_idx
            t0 = time.time()
            _, J, hist = runner(trial_seed)
            wall = time.time() - t0
            results[name]["J_values"].append(J)
            results[name]["wall_s"].append(wall)
            results[name]["histories"].append(_thin(hist))
            print(f"  trial {trial} {name:22s} seed={trial_seed} J={J:.6f} ({wall:.1f}s)")

    print("\nsummary (mean +/- std over trials, [min, max]):")
    for name, d in results.items():
        js = np.array(d["J_values"])
        d["mean"], d["std"], d["min"], d["max"] = float(js.mean()), float(js.std()), float(js.min()), float(js.max())
        print(f"  {name:22s} {d['mean']:.6f} +/- {d['std']:.6f}  [{d['min']:.6f}, {d['max']:.6f}]")

    return results


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    t0 = time.time()
    _warmup_jit()
    print(f"JIT warmup done in {time.time() - t0:.1f}s (excluded from all timed budgets below)")

    all_results = {}

    all_results["n26_fresh"] = run_experiment(26, time_budget_s=20)
    all_results["n26_warmstart"] = run_experiment(26, time_budget_s=20, warm_start=True,
                                                   label_suffix=" (warm-started from good_permutation.txt)")
    all_results["n78_fresh_multiseed"] = run_experiment_multiseed(
        78, time_budget_s=25, n_trials=5, label_suffix=" (fresh hyperplane seed)"
    )

    out_path = RESULTS_DIR / "particle_swarm_experiments.json"

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    out_path.write_text(json.dumps(_clean(all_results), indent=2))
    print(f"\nWrote raw results to {out_path}")


if __name__ == "__main__":
    main()
