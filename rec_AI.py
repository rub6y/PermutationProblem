import numpy as np
from numba import njit, prange

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

def recurence_construction(T, n):
    if n < 1:
        return T
    N = 3
    new_T = []
    for x in T:
        for i in range(N):
            new_T.append( (N*x[0] + Sym_T[i][0],
                           N*x[1] + Sym_T[i][1],
                           N*x[2] + Sym_T[i][2],
                           N*x[3] + Sym_T[i][3],
                           N*x[4] + Sym_T[i][4],
                           N*x[5] + Sym_T[i][5]) )
    return recurence_construction(new_T, n-1)

S1_T = from_perms_to_tensor(S1)
Sym_T = from_perms_to_tensor(Sym)

@njit
def get_order_code(a, b, c):
    if a < b:
        if b < c: return 0
        elif a < c: return 1
        else: return 3
    else:
        if a < c: return 2
        elif b < c: return 4
        else: return 5

@njit
def is_triple_good_fast(x1, x2, x3):
    seen_mask = 0
    unique_count = 0
    for d in range(6):
        v1, v2, v3 = x1[d], x2[d], x3[d]
        if v1 == v2 or v1 == v3 or v2 == v3:
            return False
        
        code = get_order_code(v1, v2, v3)
        bit = 1 << code
        
        if (seen_mask & bit) == 0:
            seen_mask |= bit
            unique_count += 1
            
        if unique_count < d + 1:
            return False
            
    return unique_count == 6

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
                if is_triple_good_fast(t1, t2, t3):
                    local_sum += 1
        total_good += local_sum
        
    return total_good / (M**3)

# Uruchomienie
start_T = S1_T
tensor_list = recurence_construction(start_T, 5) # Test dla n = 3
tensor_np = np.array(tensor_list, dtype=np.int64)

# Pierwsze wywołanie rozgrzewa JIT
result = compute_J_fast(tensor_np)
print(f"Wynik dla n=4: {result}")