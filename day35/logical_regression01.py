import numpy as np
import matplotlib.pyplot as plt
# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Input feature
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])   # Binary labels (0 or 1)
w = 0.5  # Weight
b = -3   # Bias
z = w * X + b
predictions = sigmoid(z)
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, predictions, color='red', label='Logistic Regression')
plt.title('Logistic Regression')
plt.xlabel('Input Feature (X)')
plt.ylabel('Prediction (Probability)')
plt.legend()
plt.grid(True)
plt.show()
