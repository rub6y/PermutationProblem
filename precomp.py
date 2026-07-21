from itertools import product
import numpy as np

N = 6
dim = 6

# All vertices of an N x N x N x N x N x N grid, represented as 6-tuples.
# This gives every point in [0, N-1]^6, i.e. one vertex for each 1x1 cube.
dim_cubes = list(product(range(N), repeat=dim))
dim_cubes_array = np.array(dim_cubes, dtype=int)
#print(dim_cubes_array)

Sym = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

def get_ordering(a, b, c):
    vals = np.array([a, b, c])
    return tuple(np.argsort(np.argsort(vals)))

def ordered_triples():
    for i in range(N):
        for j in range(i+1,N):
            for k in range(j+1,N):
                yield (i,j,k)

ordered_triples_list = list(ordered_triples())
print("DONE 1")

def permuted_triples(sigma):
    for triple in ordered_triples_list:
        yield (triple[sigma[0]], triple[sigma[1]], triple[sigma[2]])

# print(list(permuted_triples(Sym[3])))

with open('precomp.txt', 'w') as f:
    all_permuted_triples = list(product(*[list(permuted_triples(sigma)) for sigma in Sym]))
    f.write(all_permuted_triples.__str__())

print("DONE 2")



# with open('precomp.txt', 'w') as f:
#     for i in range(len(dim_cubes_array)):
#         for j in range(len(dim_cubes_array)):
#             for k in range(len(dim_cubes_array)):
#                 x1 = dim_cubes_array[i]
#                 x2 = dim_cubes_array[j]
#                 x3 = dim_cubes_array[k]
                
#                 orderings = set()
#                 for d in range(dim):
#                     if x1[d] == x2[d] or x1[d] == x3[d] or x2[d] == x3[d]:
#                         break
#                     ordering = get_ordering(x1[d], x2[d], x3[d])
#                     orderings.add(ordering)
#                     if len(orderings) < d + 1: 
#                         break
               
#                 if len(orderings) == dim:
#                     f.write(f"{x1.tolist()}, {x2.tolist()}, {x3.tolist()}\n")
