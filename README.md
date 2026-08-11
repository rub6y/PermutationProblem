# PermutationProblem

Computer search code and experiment history for the triple shattering
permutation problem. Start at `../KNOWLEDGE.md` for the compact summary
(problem statement, theory, best results, open threads).

- `EXPERIMENTS_LOG.md` — full chronological write-up of every search
  strategy tried (Frank-Wolfe, basin-hop, simulated annealing, particle
  swarm, recursive construction, symmetry search).
- `theory_approach/THEORY_NOTES.md` — theory-motivated construction attempts
  (growing-gadget/convex-position, torus reflection pairs, independence
  baseline, perturbative/LP-oracle stationarity check).
- `fw/` — core library (objective, gradient, LP oracle, symmetry).
- `scripts/` — experiment drivers.
- `results/`, `plots/` — raw output and figures.
