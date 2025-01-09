import numpy as np
from scipy import linalg

"""

x+2y-3z=-3

2x-5y+4z=13

5x+4y-z=5

"""
a = np.array([[1, 2, -3], [2, -5, 4], [5, 4, -1]])
b = np.array([[-3], [13], [5]])
x = linalg.solve(a, b)

print("Solution x:")
print(x)
# Verification
print("Verification (a * x - b):")
print(a.dot(x) - b)
