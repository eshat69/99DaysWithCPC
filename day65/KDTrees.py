from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import cosine

points = [(1, -1), (2, 3), (-2, 3), (2, -3)]

kdtree = KDTree(points)

res = kdtree.query((1, 1))

print(res)

p1 = (1, 0)
p2 = (10, 2)
res = cityblock(p1, p2)
print(res)
res1 = euclidean(p1, p2)
res = cosine(p1, p2)
print(res)
print(res1)