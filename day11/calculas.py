import numpy as np
import matplotlib.pyplot as plt

def sin(x):
    # Handles division by zero case
    return np.where(x == 0, 1, np.sin(x) / x)
x = np.linspace(-10, 10, 1000)
y = sin(x)

# Plotting
fig, ax = plt.subplots()
ax.axvline(x=0, color='lightgray')
ax.axhline(y=0, color='lightgray')
ax.axvline(x=0, color='purple', linestyle='--')
ax.axhline(y=1, color='red', linestyle='--')

# Set x and y limits
ax.set_xlim(-10, 10)
ax.set_ylim(-1, 2)

# Plot sin(x)/x
ax.plot(x, y, color="blue", label="sin(x)/x")
ax.legend()
plt.xlabel("x")
plt.ylabel("sin(x)/x")
plt.title("Plot of sin(x)/x")

plt.show()
