import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
x = sym.symbols('x')
series1 = sym.series(sym.cos(x), x, 0, 4)
print("Taylor series for cos(x):", series1)
# Convert the series to a lambda function for plotting
taylor_cos = sym.lambdify(x, series1.removeO(), "numpy")

# Set up the range for x values
x_vals = np.linspace(-2 * np.pi, 2 * np.pi, 100)

# Actual cos(x) values
y_vals_cos = np.cos(x_vals)

# Taylor series approximation values
y_vals_taylor_cos = taylor_cos(x_vals)
# Plotting
plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_vals_cos, label=r'$\cos(x)$', color='blue')
plt.plot(x_vals, y_vals_taylor_cos, label='Taylor Series of cos(x)', linestyle='--', color='red')
plt.title(r'$\cos(x)$ and its Taylor Series Approximation')
plt.xlabel('x')
plt.ylabel(r'$\cos(x)$')
plt.legend()
plt.grid(True)
plt.show()
