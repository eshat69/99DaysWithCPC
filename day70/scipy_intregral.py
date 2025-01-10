import math
from scipy.integrate import quad

def f(x):
    return math.sin(x)

result, error = quad(f, 0, math.pi/2)
print(result)

from scipy.integrate import dblquad

def f(x, y):
    return x**2 + y**2

area, error = dblquad(f, -5, 5, lambda x: -10, lambda x: 10)
print(area)

from scipy.integrate import nquad

def f(x, y, z):
    return x * y * z

result, error = nquad(f, [[0, 2], [-2, 0], [1, 2]])
print(result)
