import numpy as np
a = np.array([
    [1, 3, 5],
    [2, 6, 8],
    [3, 5, 8]
])

b = np.array([
    [2, 5, 7],
    [8, 9, 4],
    [1, 2, 3]
])
c = np.array([
    [1, 3],
    [2, 6 ]
])

d = np.array([
    [2, 5,],
    [8, 9 ]
])
e = np.array([
    [3, 3,],
    [4, 1 ]
])
print(a.dot(b))
print(np.dot(c,d))
print(np.matmul(d,e))
print(c@e)
