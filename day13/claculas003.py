import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

# Define the symbol x for symbolic operations
x = sym.symbols('x')

# Indefinite integration of cos(x) with respect to dx
integral1 = sym.integrate(sym.cos(x), x)
print('Indefinite integral of cos(x):', integral1)

# Definite integration of cos(x) with respect to dx between -1 to 1
integral2 = sym.integrate(sym.cos(x), (x, -1, 1))
print('Definite integral of cos(x) between -1 to 1:', integral2)

# Definite integration of exp(-x) with respect to dx between 0 to infinity
integral3 = sym.integrate(sym.exp(-x), (x, 0, sym.oo))
print('Definite integral of exp(-x) between 0 to ∞:', integral3)

# Plotting
x_vals_cos = np.linspace(-2 * np.pi, 2 * np.pi, 1000)  # Range for cos(x)
y_vals_cos = np.cos(x_vals_cos)

x_vals_exp = np.linspace(0, 5, 1000)  # Range for exp(-x)
y_vals_exp = np.exp(-x_vals_exp)

plt.figure(figsize=(12, 6))

# Plot for cos(x)
plt.subplot(1, 2, 1)
plt.plot(x_vals_cos, y_vals_cos, label=r'$\cos(x)$')
plt.fill_between(x_vals_cos, y_vals_cos, where=(x_vals_cos >= -1) & (x_vals_cos <= 1), alpha=0.3)
plt.title(r'Integral of $\cos(x)$ from -1 to 1')
plt.xlabel('x')
plt.ylabel(r'$\cos(x)$')
plt.legend()
plt.grid()

# Plot for exp(-x)
plt.subplot(1, 2, 2)
plt.plot(x_vals_exp, y_vals_exp, label=r'$e^{-x}$')
plt.fill_between(x_vals_exp, y_vals_exp, alpha=0.3)
plt.title(r'Integral of $e^{-x}$ from 0 to ∞')
plt.xlabel('x')
plt.ylabel(r'$e^{-x}$')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
