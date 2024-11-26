import matplotlib.pyplot as plt
import numpy as np
from sympy import Symbol, lambdify
x = Symbol('x')
f = x**3 - 2*x - 5
f_prime = f.diff(x)
f = lambdify(x, f)
f_prime = lambdify(x, f_prime)
# Input for the Newton-Raphson Method
x_value = float(input("Enter the initial guess for x: "))
n = int(input("Enter the required correct decimal places: "))
# Newton-Raphson Method
i = 1
while True:
    prev_x = x_value
    x_value = x_value - (f(prev_x) / f_prime(prev_x))  # Newton-Raphson formula
    print(f"Iteration {i}: x = {x_value:.{n + 2}f}, f(x) = {f(x_value):.{n + 2}f}")
    if round(prev_x, n) == round(x_value, n):
        break
    i += 1
print(f"The root is approximately: {x_value:.{n}f} after {i} iterations")
m = np.linspace(-3, 3, 100)
y = f(m)
# Plot the function
plt.plot(m, y, label='f(x)')
plt.axhline(0, color='red', linestyle='--', label='y=0')
plt.scatter([x_value], [f(x_value)], color='blue', label=f'Root ≈ {x_value:.{n}f}')
plt.title("Newton Raphson Method  Plot of f(x) and its Root")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()
