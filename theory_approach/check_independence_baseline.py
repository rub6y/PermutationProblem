"""Quick sanity check: what if the 6 permutations are made deliberately
UNcorrelated (pseudorandom / independent), instead of forced onto H like
every construction tried so far?

Two variants:
1. True iid random permutations of [n] (numpy shuffle x6).
2. "Discrete-log" pseudorandom family: for a prime p, g a primitive root,
   sigma_k(t) = index of g^{a_k * t} mod p among {1,...,p-1}, for 6 distinct
   exponents a_k coprime to p-1. This is a genuinely different generative
   mechanism from every affine/recursive family tried (multiplicative, not
   additive), included here because IF it behaved like a "structured but
   still fairly independent" family it might have looked promising -- this
   checks whether that intuition has any merit before spending more effort
   on it.

Both are compared against the theoretical iid-independence expectation
6!/6^6 = 720/46656 ~ 0.01543 (probability 6 iid uniform order-types cover
all of S_3), to see whether real finite-n permutations already reach that
asymptotic or overshoot/undershoot it.
"""

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fw.shattering import count_shattered_triples  # noqa: E402


def _primitive_root(p):
    n = p - 1
    factors = set()
    m = n
    d = 2
    while d * d <= m:
        while m % d == 0:
            factors.add(d)
            m //= d
        d += 1
    if m > 1:
        factors.add(m)
    for g in range(2, p):
        if all(pow(g, n // f, p) != 1 for f in factors):
            return g
    raise ValueError("no primitive root found")


def iid_random_J(n, seed=0):
    rng = np.random.default_rng(seed)
    points = np.stack([rng.permutation(n) for _ in range(6)], axis=1).astype(np.int64)
    return count_shattered_triples(points) / n**3


def discrete_log_J(p, exponents):
    g = _primitive_root(p)
    n = p - 1
    # x_k(t) = g^(a_k * t) mod p for t = 0..n-1; rank = value itself (already
    # a bijection [0,n-1] -> {1,...,p-1}, so its natural integer order IS the
    # permutation we need -- no separate ranking step required).
    powers_of_g = np.empty(n, dtype=np.int64)
    val = 1
    for i in range(n):
        powers_of_g[i] = val
        val = (val * g) % p
    # powers_of_g[i] = g^i mod p
    points = np.empty((n, 6), dtype=np.int64)
    for k, a in enumerate(exponents):
        idx = (a * np.arange(n)) % n
        col = powers_of_g[idx]
        points[:, k] = np.argsort(np.argsort(col))
    return count_shattered_triples(points) / n**3


if __name__ == "__main__":
    theory = 720 / 6**6
    print(f"theoretical iid-independence limit: {theory:.6f}")
    for n in (26, 60, 100):
        print(f"iid random permutations, n={n}: J = {iid_random_J(n):.6f}")

    p = 101  # n = 100, prime
    exps = [1, 3, 7, 13, 17, 23]  # arbitrary distinct exponents coprime to 100
    print(f"discrete-log family, p={p} (n={p-1}), exponents={exps}: "
          f"J = {discrete_log_J(p, exps):.6f}")
