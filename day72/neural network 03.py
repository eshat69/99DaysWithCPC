import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Sample dataset (binary classification)
data = {
    'feature1': [0.1, 0.2, 0.3, 0.4, 0.5],
    'feature2': [0.5, 0.4, 0.3, 0.2, 0.1],
    'label': [0, 0, 1, 1, 1]  # Binary labels
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features (X) and Labels (y)
X = df[['feature1', 'feature2']].values  # Input features
y = df['label'].values  # Output labels

# Create a Sequential model
model = Sequential()

# Add input layer and hidden layer
model.add(Dense(8, input_dim=2, activation='relu'))  # 2 input features, 8 neurons in hidden layer

# Add output layer
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=100, batch_size=1, verbose=1)

# Example input for prediction
test_data = np.array([[0.2, 0.4]])  # New input data

# Make a prediction
prediction = model.predict(test_data)
predicted_label = (prediction > 0.5).astype(int)  # Convert probability to binary output
print(f"Predicted label: {predicted_label[0][0]}")