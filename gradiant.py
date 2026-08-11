import numpy as np
import random
from itertools import permutations, combinations

from fw.shattering import shattering_reference as is_triple_good

N = 26

S1 = [
  (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25),
  (18, 17, 19, 5, 4, 3, 24, 25, 23, 16, 13, 14, 10, 15, 11, 12, 9, 1, 0, 2, 22, 21, 20, 7, 6, 8),
  (18, 19, 17, 21, 20, 22, 8, 7, 6, 13, 16, 11, 15, 9, 12, 10, 14, 1, 2, 0, 4, 3, 5, 25, 24, 23),
  (21, 18, 17, 10, 14, 13, 9, 5, 6, 3, 24, 23, 1, 22, 0, 25, 2, 20, 16, 19, 12, 15, 11, 4, 8, 7),
  (2, 0, 1, 21, 22, 20, 24, 23, 25, 10, 9, 16, 15, 14, 13, 12, 11, 19, 18, 17, 3, 5, 4, 7, 8, 6),
  (16, 20, 19, 15, 11, 12, 4, 8, 7, 24, 3, 0, 22, 2, 25, 1, 23, 17, 21, 18, 14, 10, 13, 9, 5, 6)
] 

def sample_gen(sample_count = 200):
    sample_set = []
    for _ in range(sample_count):
        x1 = random.random()
        x2 = random.random()
        x3 = 1.5 - x1 - x2 
        x4 = random.random()
        x5 = random.random()
        x6 = 1.5 - x4 - x5
        if (x6 > 0 and x6 < 1) and (x3 > 0 and x3 < 1):
            sample_set.append((x1,x2,x3,x4,x5,x6))
        
    return sample_set

genereted_samples = sample_gen()

def triples_count(samples):
    count = 0
    for x1 in samples:
        for x2 in samples:
            for x3 in samples:
                count += is_triple_good(x1,x2,x3)
    M = len(samples)
    normalizer = (M * (M-1)*(M-2))/6  
    return (count/normalizer, len(samples))

answer = triples_count(genereted_samples)

with open('gradient.txt', 'w') as f:
    f.write(answer.__str__())
    f.write("\n")
    f.write(genereted_samples.__str__())

"""
S2 = random.sample(S1, len(S1))
S3 = random.sample(S1, len(S1))
S4 = random.sample(S1, len(S1))
S5 = random.sample(S1, len(S1))
S6 = random.sample(S1, len(S1))
S7 = random.sample(S1, len(S1))
S8 = random.sample(S1, len(S1))
S9 = random.sample(S1, len(S1))
S10 = random.sample(S1, len(S1))

def from_perms_to_tensor(perms):
    tensor = []
    for i in range(N):
        tmp_pair = ((perms[0][i], perms[1][i], perms[2][i], perms[3][i], perms[4][i], perms[5][i]), 1)
        tensor.append(tmp_pair)
    return tensor


def compute_J_from_tensor(tensor):
    sum = 0
    for t1 in tensor:
        for t2 in tensor:
            for t3 in tensor:
                if is_triple_good(t1[0], t2[0], t3[0]):
                    sum += t1[1] * t2[1] * t3[1]
    return sum/(N**3)

T1 = from_perms_to_tensor(S1)
T2 = from_perms_to_tensor(S2)
T3 = from_perms_to_tensor(S3)
T4 = from_perms_to_tensor(S4)
T5 = from_perms_to_tensor(S5)
T6 = from_perms_to_tensor(S6)
T7 = from_perms_to_tensor(S7)
T8 = from_perms_to_tensor(S8)
T9 = from_perms_to_tensor(S9)
T10 = from_perms_to_tensor(S10)

T = T2
print(S2)

print(compute_J_from_tensor(T))
"""