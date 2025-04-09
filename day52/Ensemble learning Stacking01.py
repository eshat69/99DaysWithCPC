from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier


# Load dataset (example with Iris dataset)
data = load_iris()
X, y = data.data, data.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base classifiers
base_classifiers = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42))
]

# Define meta-classifier
meta_classifier = LogisticRegression(random_state=42)

# Create stacking classifier
stacking_clf = StackingClassifier(estimators=base_classifiers, final_estimator=meta_classifier)

# Train stacking classifier
stacking_clf.fit(X_train, y_train)

# Predict using stacking classifier
y_pred = stacking_clf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Stacking Classifier Accuracy: {accuracy:.2f}")
