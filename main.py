import numpy as np
import random
from itertools import permutations, combinations

S3 = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

S_list = []
for p in permutations(S3, 6):
    S_list.append(p)

n = 10000
dim = 6

def get_ordering(a, b, c):
    vals = np.array([a, b, c])
    return tuple(np.argsort(np.argsort(vals)))

def generate_random_permutations(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    sigma = np.zeros((6, n), dtype=int)
    for k in range(6):
        sigma[k] = np.random.permutation(n) + 1
    
    return sigma

sigma_rand = generate_random_permutations(n, seed=random.randint(0, 10000))

def compute_J_Monte_Carlo(sigma, num_samples = 50000):
    count = 0
    for _ in range(num_samples):
        idx = np.random.choice(n, 3, replace=False)
        i, j, k = idx
        
        x1 = (sigma[:, i] - 1.0 + np.random.rand(dim)) / n
        x2 = (sigma[:, j] - 1.0 + np.random.rand(dim)) / n
        x3 = (sigma[:, k] - 1.0 + np.random.rand(dim)) / n
        # x1 = np.random.rand(dim)
        # x2 = np.random.rand(dim) 
        # x3 = np.random.rand(dim)
        
        # print(x1)
        # print(x2)
        # print(x3)
        orderings = set()
        for d in range(dim):
            ordering = get_ordering(x1[d], x2[d], x3[d])
            orderings.add(ordering)
            if len(orderings) < d + 1: 
                break
       
        if len(orderings) == 6:
            count += 1

    return count / num_samples


max_J = 0.0
for _ in range(100):
    sigma_rand = generate_random_permutations(n, seed=random.randint(0, 10000))
    J_estimate = compute_J_Monte_Carlo(sigma_rand)
    if J_estimate > max_J:
        print(f"Estimated J: {J_estimate} exceeds maximum J: {max_J}")
        max_J = J_estimate

print(max_J)