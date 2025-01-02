import numpy as np
from scipy.sparse import csr_matrix

arr = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 2]])

print(csr_matrix(arr).data)
print(csr_matrix(arr).count_nonzero())
mat = csr_matrix(arr)
mat.eliminate_zeros()
print(mat)