import numpy as np
from numba import njit, prange

from fw.shattering import is_shattered_triple

S1 = [
  (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25),
  (18, 17, 19, 5, 4, 3, 24, 25, 23, 16, 13, 14, 10, 15, 11, 12, 9, 1, 0, 2, 22, 21, 20, 7, 6, 8),
  (18, 19, 17, 21, 20, 22, 8, 7, 6, 13, 16, 11, 15, 9, 12, 10, 14, 1, 2, 0, 4, 3, 5, 25, 24, 23),
  (21, 18, 17, 10, 14, 13, 9, 5, 6, 3, 24, 23, 1, 22, 0, 25, 2, 20, 16, 19, 12, 15, 11, 4, 8, 7),
  (2, 0, 1, 21, 22, 20, 24, 23, 25, 10, 9, 16, 15, 14, 13, 12, 11, 19, 18, 17, 3, 5, 4, 7, 8, 6),
  (16, 20, 19, 15, 11, 12, 4, 8, 7, 24, 3, 0, 22, 2, 25, 1, 23, 17, 21, 18, 14, 10, 13, 9, 5, 6)
] 

Sym = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

def from_perms_to_tensor(perms):
    N = len(perms[0])
    tensor = []
    for i in range(N):
        tmp = (perms[0][i], perms[1][i], perms[2][i], perms[3][i], perms[4][i], perms[5][i])
        tensor.append(tmp)
    return tensor

def recurence_construction(T, n, digits=None):
    """Base-`len(digits)` digit expansion x -> N*x + digits[i], depth `n`.

    `digits` defaults to `Sym_T[:3]` (the original 3-digit construction).
    Any digit set whose vectors all sum to the same constant preserves the
    central hyperplane H algebraically (see rec_construction_scaling.md).
    """
    if digits is None:
        digits = Sym_T[:3]
    if n < 1:
        return T
    N = len(digits)
    new_T = []
    for x in T:
        for d in digits:
            new_T.append( (N*x[0] + d[0],
                           N*x[1] + d[1],
                           N*x[2] + d[2],
                           N*x[3] + d[3],
                           N*x[4] + d[4],
                           N*x[5] + d[5]) )
    return recurence_construction(new_T, n-1, digits)

S1_T = from_perms_to_tensor(S1)
Sym_T = from_perms_to_tensor(Sym)

@njit(parallel=True)
def compute_J_fast(tensor_np):
    M = tensor_np.shape[0]
    total_good = 0

    for i in prange(M):
        t1 = tensor_np[i]
        local_sum = 0
        for j in range(M):
            t2 = tensor_np[j]
            for k in range(M):
                t3 = tensor_np[k]
                if is_shattered_triple(t1, t2, t3):
                    local_sum += 1
        total_good += local_sum

    return total_good / (M**3)

if __name__ == "__main__":
    start_T = S1_T
    tensor_list = recurence_construction(start_T, 5)
    tensor_np = np.array(tensor_list, dtype=np.int64)
    result = compute_J_fast(tensor_np)
    print(f"depth=5, M={len(tensor_list)}, J={result}")