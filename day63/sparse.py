import numpy as np
from scipy.sparse import csr_matrix

arr = np.array([0, 0, 0, 0, 0, 1, 1, 0, 2])

print(csr_matrix(arr))

mat = csr_matrix(arr)
mat.sum_duplicates()

print(mat)

mat = csr_matrix(arr)
mat.eliminate_zeros()

print(mat)