# Python code to find Euclidean distance
# using sum() and square()

import numpy as np

# initializing points in
# numpy arrays
point1 = np.array((1, 2, 3))
point2 = np.array((1, 1, 1))

# finding sum of squares
sum_sq = np.sum(np.square(point1 - point2))

# Doing squareroot and
# printing Euclidean distance
print(np.sqrt(sum_sq))