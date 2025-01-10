import numpy as np
from scipy.optimize import minimize

def objective(x):
    a = x[0]
    b = x[1]
    return a**2 + b**2

def constraint01(x):
    a = x[0]
    b = x[1]
    return a + b - 100

def constraint02(x):
    a = x[0]
    b = x[1]
    return a * b - 100

cons = [
    {'type': 'eq', 'fun': constraint01},
    {'type': 'ineq', 'fun': constraint02}
]
bound = np.array([10,50])
bounds = (bound,bound)
x0 = np.array([200, 500])
sol = minimize(objective, x0, method='SLSQP', constraints=cons,bounds=bounds)

print(sol)
