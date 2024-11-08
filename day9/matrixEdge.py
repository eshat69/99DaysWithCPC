import numpy as np
a = np.array([
    [1, 6, 9],
    [2, 45, 10],
    [5, 7, 9]
])
# Compute eigenvalues
print(   np.linalg.eigvals(a) )


arr = np.array([41, 42, 43, 44])
x = [True, False, True, False]
a = arr[x]
print(a)
