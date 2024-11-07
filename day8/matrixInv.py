import numpy as np
A= np.array([
    [2,5,7],
    [1,2,3],
    [7,8,2]
])
print(  np.linalg.inv(A))
b = np.array([
    [0,1,2],
    [1,2,3],
    [3,1,1]
])
try:
    b_inv = np.linalg.inv(b)
    print("Inverse of matrix b:\n", b_inv)
except np.linalg.LinAlgError:
    print("Matrix b is not invertible.")
