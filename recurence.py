import numpy as np
import random
from itertools import permutations, combinations


S1 = [
  (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25),
  (18, 17, 19, 5, 4, 3, 24, 25, 23, 16, 13, 14, 10, 15, 11, 12, 9, 1, 0, 2, 22, 21, 20, 7, 6, 8),
  (18, 19, 17, 21, 20, 22, 8, 7, 6, 13, 16, 11, 15, 9, 12, 10, 14, 1, 2, 0, 4, 3, 5, 25, 24, 23),
  (21, 18, 17, 10, 14, 13, 9, 5, 6, 3, 24, 23, 1, 22, 0, 25, 2, 20, 16, 19, 12, 15, 11, 4, 8, 7),
  (2, 0, 1, 21, 22, 20, 24, 23, 25, 10, 9, 16, 15, 14, 13, 12, 11, 19, 18, 17, 3, 5, 4, 7, 8, 6),
  (16, 20, 19, 15, 11, 12, 4, 8, 7, 24, 3, 0, 22, 2, 25, 1, 23, 17, 21, 18, 14, 10, 13, 9, 5, 6)
] 

Sym = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

def get_ordering(a, b, c):
    vals = np.array([a, b, c])
    return tuple(np.argsort(np.argsort(vals)))


def is_triple_good(x1, x2, x3):
    orderings = set()
    for d in range(6):
        if x1[d] == x2[d] or x1[d] == x3[d] or x2[d] == x3[d]:
            break
        ordering = get_ordering(x1[d], x2[d], x3[d])
        orderings.add(ordering)
        if len(orderings) < d + 1: 
            break
    return len(orderings) == 6

def from_perms_to_tensor(perms):
    N = len(perms[0])
    tensor = []
    for i in range(N):
        tmp = (perms[0][i], perms[1][i], perms[2][i], perms[3][i], perms[4][i], perms[5][i])
        tensor.append(tmp)
    return tensor

S1_T = from_perms_to_tensor(S1)
print(S1_T)

Sym_T = from_perms_to_tensor(Sym)
print(Sym_T)

def recurence_construction(T, n):
    if n < 1:
        print("Done")
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

# def recurence_construction2(T, n):
#     if n < 1:
#         return T

#     new_T = []
#     for x in T:
#         for i in range(3):
#             new_T.append( (3*x[0] + prime_T[i][1],
#                            3*x[1] + prime_T[i][0],
#                            3*x[2] + prime_T[i][2],
#                            3*x[3] + prime_T[i][5],
#                            3*x[4] + prime_T[i][3],
#                            3*x[5] + prime_T[i][4]) )
#     return recurence_construction(new_T, n-1)

def compute_J_from_tensor(tensor):
    sum = 0
    M = len(tensor)
    for t1 in tensor:
        for t2 in tensor:
            for t3 in tensor:
                if is_triple_good(t1, t2, t3):
                    sum += 1
    return sum/(M**3)


start_T = S1_T
print(compute_J_from_tensor(recurence_construction(start_T, 2)))

# calc_count = 5

# for i in range(1, calc_count + 1):
#     print(compute_J_from_tensor(recurence_construction(start_T, i)))
