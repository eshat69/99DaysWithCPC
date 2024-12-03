import numpy as np
from sklearn.linear_model import LogisticRegression

# Dataset represented as a dictionary
data = {
    "test_score": [85, 60, 75, 95, 50, 45, 70, 80, 90, 55],
    "admitted": [1, 0, 1, 1, 0, 0, 1, 1, 1, 0]
}

# Extract features and labels
X = np.array(data["test_score"]).reshape(-1, 1)  # Feature: test scores
y = np.array(data["admitted"])  # Labels: admitted or not
model = LogisticRegression()
model.fit(X, y)

# Make predictions
test_scores = [[65], [78], [50]]
predictions = model.predict(test_scores)
probabilities = model.predict_proba(test_scores)

# Display results
for score, pred, prob in zip(test_scores, predictions, probabilities):
    print(f"Test Score: {score[0]}, Predicted Admission: {pred}, Probability: {prob[1]:.2f}")

# Model coefficients
print("\nModel Coefficient (Weight):", model.coef_[0][0])
print("Model Intercept (Bias):", model.intercept_[0])
