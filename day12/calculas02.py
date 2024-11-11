import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

# Define symbolic variables and expression
x, y, z = sym.symbols('x y z')
exp = x**3 * y + y**3 + z
# Differentiate exp with respect to x and y
derivative1_x = sym.diff(exp, x)
print('Derivative with respect to x:', derivative1_x)

derivative1_y = sym.diff(exp, y)
print('Derivative with respect to y:', derivative1_y)
x_vals = np.linspace(-10, 10, 1000)
y_vals = np.sinc(x_vals / np.pi)  # sinc(x) in numpy is sin(x)/x

# Plotting
fig, ax = plt.subplots()
ax.axvline(x=0, color='lightgray')
ax.axhline(y=0, color='lightgray')
ax.axvline(x=0, color='yellow', linestyle='--')
ax.axhline(y=1, color='green', linestyle='--')
# Set x and y limits
ax.set_xlim(-10, 10)
ax.set_ylim(-1, 2)
# Plot sin(x)/x
ax.plot(x_vals, y_vals, color="black", label="sin(x)/x")
ax.legend()
plt.xlabel("x")
plt.ylabel("sin(x)/x")
plt.title("Plot of sin(x)/x")
plt.show()
