import numpy as np
from collections import Counter


# Define a function to calculate Euclidean distance between two points
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
# Define the k-Nearest Neighbors algorithm
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


# Example usage:
if __name__ == "__main__":
    X_train = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    y_train = np.array([0, 0, 1, 1, 0, 1])

    X_test = np.array([[1, 3], [4, 5], [3, 7], [6, 10]])

    # Initialize k-NN classifier
    clf = KNN(k=3)
    clf.fit(X_train, y_train)

    # Predictions on the test data
    predictions = clf.predict(X_test)
    print("Predictions:", predictions)
