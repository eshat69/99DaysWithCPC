# Python code to find Euclidean distance
# using dot()

import numpy as np

# initializing points in
# numpy arrays
point1 = np.array((1, 2, 3))
point2 = np.array((1, 1, 1))

# subtracting vector
temp = point1 - point2

# doing dot product
# for finding
# sum of the squares
sum_sq = np.dot(temp.T, temp)

# Doing squareroot and
# printing Euclidean distance
print(np.sqrt(sum_sq))