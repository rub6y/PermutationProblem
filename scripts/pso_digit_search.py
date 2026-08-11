"""PSO over the *digit-set* of rec_AI.py's recursive construction, not over
any fixed-n point cloud -- the search space (N digit vectors in Z^6, each
summing to the same constant C so the hyperplane invariant is preserved
recursively, per rec_AI.recurence_construction's docstring) has fixed
dimension regardless of how large n eventually gets built up to.

Context: records.jsonl's best-ever witness (M=1080, J=0.497493) is
`a_summary-n120v2-recursion_depth2` -- rec_AI.recurence_construction applied
twice to the best n=120 witness, using a hand-picked digit-set
(Sym_T[:3], three rows of the S_3 permutation tensor). Nobody has optimized
that digit-set itself. The real, larger search space -- 6 integers in
[0, N-1] summing to C=3, of which the 6 S_3-permutation rows are only a
small subset -- is what PSO searches here.

Fitness is evaluated cheaply (depth=1 off good_permutation.txt, M=26->78)
for many PSO iterations; the best candidates found are then re-validated at
depth=1 off the M=120 a_summary seed and at depth=2, since a digit-set that
looks good at the cheap depth might not generalize (NEXT_recurrence_scaling_plan.md
already documented per-depth gains shrinking).

Multimodality (many isolated local optima, cf. records_structure_m78.md
documenting structurally distinct near-optimal families): each island
occasionally takes a large random jump instead of a small velocity step
(mirrors records.jsonl's already-validated basin_hop-large_perturb result),
and several independent islands run with rare migration instead of one swarm.

Usage: nix-shell shell.nix --run "python3 scripts/pso_digit_search.py"
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from fw.measure import load_permutations
from fw.shattering import count_shattered_triples
from rec_AI import Sym_T, recurence_construction

ROOT = Path(__file__).resolve().parent.parent
GOOD_PERM_PATH = ROOT.parent / "good_permutation.txt"
RECORDS_PATH = ROOT.parent / "records.jsonl"
RESULTS_DIR = ROOT / "results"

N_DIGITS = 3
C_TARGET = int(sum(Sym_T[0]))  # sum every digit vector must hit; Sym_T's rows sum to 6, not N_DIGITS
VAL_MAX = N_DIGITS - 1  # each of the 6 coords lives in [0, VAL_MAX]


def _warmup_jit():
    dummy = np.array([[0, 1, 2, 0, 1, 2], [2, 1, 0, 2, 1, 0], [1, 2, 0, 1, 0, 2]], dtype=np.int64)
    count_shattered_triples(dummy)


def random_digit(rng, max_tries=2000):
    """A random length-6 int vector in [0, VAL_MAX]^6 summing to C_TARGET,
    via rejection sampling (search space is tiny, this is fast)."""
    for _ in range(max_tries):
        v = rng.integers(0, VAL_MAX + 1, size=6)
        if v.sum() == C_TARGET:
            return v.astype(np.float64)
    return np.array(Sym_T[0], dtype=np.float64)  # fallback: a known-valid digit


def random_digitset(rng):
    return np.stack([random_digit(rng) for _ in range(N_DIGITS)])


def project_digitset(x):
    """Round to nearest int, clip to [0, VAL_MAX], then nudge coordinates
    (away from their bound, in the needed direction) until each digit's sum
    is exactly C_TARGET again."""
    v = np.clip(np.round(x), 0, VAL_MAX).astype(np.int64)
    for d in range(v.shape[0]):
        diff = C_TARGET - int(v[d].sum())
        guard = 0
        while diff != 0 and guard < 200:
            guard += 1
            coord = np.random.randint(0, 6)
            if diff > 0 and v[d, coord] < VAL_MAX:
                v[d, coord] += 1
                diff -= 1
            elif diff < 0 and v[d, coord] > 0:
                v[d, coord] -= 1
                diff += 1
    return v


def fitness(digitset_int, seed_tensor, depth=1):
    digits = [tuple(row) for row in digitset_int.tolist()]
    expanded = recurence_construction(seed_tensor, depth, digits=digits)
    pts = np.array(expanded, dtype=np.int64)
    sums = pts.sum(axis=1)
    if not np.all(sums == sums[0]):
        return -1.0, pts  # constraint violated -- shouldn't happen post-projection, but guard anyway
    M = pts.shape[0]
    return count_shattered_triples(pts) / M**3, pts


def pso_islands(
    seed_tensor,
    n_islands=4,
    particles_per_island=6,
    n_iters=150,
    w=0.6,
    c1=1.4,
    c2=1.4,
    big_jump_prob=0.08,
    migrate_every=25,
    seed=0,
):
    rng = np.random.default_rng(seed)
    dim_shape = (N_DIGITS, 6)

    islands = []
    for _ in range(n_islands):
        pos = [random_digitset(rng).astype(np.float64) for _ in range(particles_per_island)]
        vel = [np.zeros(dim_shape) for _ in range(particles_per_island)]
        pbest_pos = [p.copy() for p in pos]
        pbest_val = [fitness(project_digitset(p), seed_tensor)[0] for p in pos]
        best_i = int(np.argmax(pbest_val))
        gbest_pos, gbest_val = pbest_pos[best_i].copy(), pbest_val[best_i]
        islands.append(dict(pos=pos, vel=vel, pbest_pos=pbest_pos, pbest_val=pbest_val,
                             gbest_pos=gbest_pos, gbest_val=gbest_val))

    global_best_pos = max(islands, key=lambda isl: isl["gbest_val"])["gbest_pos"].copy()
    global_best_val = max(isl["gbest_val"] for isl in islands)
    history = [(0, global_best_val)]

    for it in range(1, n_iters + 1):
        for isl in islands:
            for p in range(particles_per_island):
                if rng.random() < big_jump_prob:
                    isl["pos"][p] = random_digitset(rng).astype(np.float64)
                    isl["vel"][p] = np.zeros(dim_shape)
                else:
                    r1, r2 = rng.random(dim_shape), rng.random(dim_shape)
                    isl["vel"][p] = (w * isl["vel"][p]
                                      + c1 * r1 * (isl["pbest_pos"][p] - isl["pos"][p])
                                      + c2 * r2 * (isl["gbest_pos"] - isl["pos"][p]))
                    isl["pos"][p] = isl["pos"][p] + isl["vel"][p]

                val, _ = fitness(project_digitset(isl["pos"][p]), seed_tensor)
                if val > isl["pbest_val"][p]:
                    isl["pbest_val"][p] = val
                    isl["pbest_pos"][p] = isl["pos"][p].copy()
                if val > isl["gbest_val"]:
                    isl["gbest_val"] = val
                    isl["gbest_pos"] = isl["pos"][p].copy()

        if it % migrate_every == 0:
            best_isl = max(islands, key=lambda isl: isl["gbest_val"])
            for isl in islands:
                if isl is not best_isl:
                    worst_p = int(np.argmin(isl["pbest_val"]))
                    isl["pos"][worst_p] = best_isl["gbest_pos"].copy()
                    isl["pbest_pos"][worst_p] = best_isl["gbest_pos"].copy()
                    isl["pbest_val"][worst_p] = best_isl["gbest_val"]

        cur_best_val = max(isl["gbest_val"] for isl in islands)
        if cur_best_val > global_best_val:
            global_best_val = cur_best_val
            global_best_pos = max(islands, key=lambda isl: isl["gbest_val"])["gbest_pos"].copy()
        history.append((it, global_best_val))

    return project_digitset(global_best_pos), global_best_val, history


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    _warmup_jit()
    print(f"JIT warmup: {time.time() - t0:.1f}s")

    perms = load_permutations(GOOD_PERM_PATH)
    seed_pts = np.array(perms, dtype=np.int64).T
    seed_tensor = [tuple(row) for row in seed_pts.tolist()]
    baseline_digits = Sym_T[:N_DIGITS]  # the hand-picked scheme this session is trying to beat
    baseline_val, _ = fitness(np.array(baseline_digits), seed_tensor, depth=1)
    print(f"baseline digits (Sym_T[:3]) depth=1 off M=26 seed: J={baseline_val:.6f}")

    n_runs = 4
    all_runs = []
    for run in range(n_runs):
        t0 = time.time()
        best_digits, best_val, hist = pso_islands(seed_tensor, seed=run)
        wall = time.time() - t0
        print(f"run {run}: best J={best_val:.6f}  digits={best_digits.tolist()}  ({wall:.1f}s)")
        all_runs.append({"seed": run, "J": float(best_val), "digits": best_digits.tolist(),
                          "wall_s": wall, "history": hist[::5]})

    best_run = max(all_runs, key=lambda r: r["J"])
    print(f"\nbest of {n_runs} PSO runs: J={best_run['J']:.6f} vs baseline {baseline_val:.6f} "
          f"(gain {best_run['J'] - baseline_val:+.6f})")

    # Validate the best found digit-set at the depths that actually matter:
    # depth=1 and depth=2 off the M=120 a_summary seed (the one the current
    # M=1080 record was built from), not just the cheap M=26 search seed.
    with open(RECORDS_PATH) as f:
        records = [json.loads(line) for line in f]
    m120_candidates = [r for r in records if r["M"] == 120]
    m120_best = max(m120_candidates, key=lambda r: r["J"])
    m120_pts = np.array(m120_best["perms"], dtype=np.int64).T
    m120_tensor = [tuple(row) for row in m120_pts.tolist()]

    found_digits = best_run["digits"]
    for depth in (1, 2):
        val_found, pts_found = fitness(np.array(found_digits), m120_tensor, depth=depth)
        val_base, pts_base = fitness(np.array(baseline_digits), m120_tensor, depth=depth)
        print(f"M=120 seed, depth={depth}: found-digits J={val_found:.6f}  "
              f"Sym_T[:3] J={val_base:.6f}  (M={pts_found.shape[0]})")

    out = {"baseline_J_M26_depth1": float(baseline_val), "runs": all_runs,
           "m120_seed_opt": m120_best["params"].get("opt"), "m120_seed_J": m120_best["J"]}
    (RESULTS_DIR / "pso_digit_search.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote results/pso_digit_search.json")


if __name__ == "__main__":
    main()
