from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate some synthetic data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize base models
model1 = DecisionTreeClassifier(random_state=42)
model2 = SVC(kernel='linear', random_state=42)

# Create an ensemble using VotingClassifier
ensemble_model = VotingClassifier(estimators=[
    ('dt', model1),
    ('svc', model2)
], voting='hard')  # 'hard' voting for classification, 'soft' for probability weighted voting

# Train ensemble model
ensemble_model.fit(X_train, y_train)

# Make predictions
y_pred = ensemble_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Ensemble Accuracy: {accuracy:.2f}')

