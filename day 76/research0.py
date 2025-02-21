import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import cv2
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score


# Define your image dimensions and batch size
img_height, img_width = 224, 224  # Update to ResNet50's expected input shape
batch_size = 32

# Define your data directory
data_dir = '/content/drive/MyDrive/animal'

# Assuming 'labels' is a list containing the class labels
labels = os.listdir(data_dir)  # Get the labels from directory
num_classes = len(labels)

# Function to load images and labels
def load_data(directory):
    X = []
    Y = []

    for label_idx, label in enumerate(labels):
        folderpath = os.path.join(directory, label)

        if os.path.exists(folderpath) and os.path.isdir(folderpath):
            for file in os.listdir(folderpath):
                img_path = os.path.join(folderpath, file)
                img = cv2.imread(img_path)

                if img is not None:
                    img = cv2.resize(img, (img_height, img_width))
                    X.append(np.array(img))
                    Y.append(label_idx)

    return np.array(X), np.array(Y)

# Load training and validation data
X, Y = load_data(data_dir)

# Split data into train and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
Y_train_one_hot = to_categorical(Y_train, num_classes=num_classes)
Y_val_one_hot = to_categorical(Y_val, num_classes=num_classes)

# Load ResNet50 base model without top layer and set weights to 'imagenet'
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Add layers on top of ResNet50
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # Add more dense layers if needed
predictions = Dense(num_classes, activation='softmax')(x)

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model
history = model.fit(X_train, Y_train_one_hot, epochs=15, validation_data=(X_val, Y_val_one_hot))

# Predict on the whole dataset
Y_pred = model.predict(X)
Y_pred_classes = np.argmax(Y_pred, axis=1)

# Evaluate the model
acc_score = accuracy_score(Y, Y_pred_classes)
print("Accuracy Score:", acc_score)

# Confusion matrix
conf_matrix = confusion_matrix(Y, Y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Affichage de l'historique de perte
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss History')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Affichage de l'historique de précision
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy History')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

report = classification_report(Y, Y_pred_classes, target_names=labels, labels=np.unique(Y))
print(report)

import matplotlib.pyplot as plt
import numpy as np

# Example data from your classification report
labels = ['Cheetah', 'Jaguar', 'Leopard', 'Tiger', 'Lion']
precision = [0.86, 0.87, 0.87, 1.00, 1.00]
recall = [0.97, 0.93, 0.67, 1.00, 1.00]
f1_score = [0.91, 0.90, 0.75, 1.00, 1.00]

# Calculate positions for bars
bar_width = 0.2
index = np.arange(len(labels))
index_precision = index - bar_width
index_recall = index
index_f1_score = index + bar_width

# Plotting
plt.figure(figsize=(12, 8))  # Increase figure size

bars1 = plt.bar(index_precision, precision, width=bar_width, label='Precision', color='blue')
bars2 = plt.bar(index_recall, recall, width=bar_width, label='Recall', color='green')
bars3 = plt.bar(index_f1_score, f1_score, width=bar_width, label='F1-Score', color='orange')

import matplotlib.pyplot as plt
import numpy as np

# Example data from your classification report
labels = ['Cheetah', 'Jaguar', 'Leopard', 'Tiger', 'Lion']
precision = [0.86, 0.87, 0.87, 1.00, 1.00]
recall = [0.97, 0.93, 0.67, 1.00, 1.00]
f1_score = [0.91, 0.90, 0.75, 1.00, 1.00]

# Calculate positions for bars
bar_width = 0.2
index = np.arange(len(labels))
index_precision = index - bar_width
index_recall = index
index_f1_score = index + bar_width

# Plotting
plt.figure(figsize=(12, 8))  # Increase figure size

bars1 = plt.bar(index_precision, precision, width=bar_width, label='Precision', color='blue')
bars2 = plt.bar(index_recall, recall, width=bar_width, label='Recall', color='green')
bars3 = plt.bar(index_f1_score, f1_score, width=bar_width, label='F1-Score', color='orange')

# Adding values on top of bars with bold formatting for precision, recall, and f1-score
for bars, values in zip([bars1, bars2, bars3], [precision, recall, f1_score]):
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{value:.2f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

# Adjusting plot aesthetics
plt.ylim([0, 1.2])  # Adjust y-axis limit for better visualization of values
plt.xlabel('Class', fontsize=12, fontweight='bold')  # Bold and larger x-axis label
plt.ylabel('Score', fontsize=12, fontweight='bold')  # Bold and larger y-axis label
plt.title('Classification Metrics', fontsize=16, fontweight='bold')  # Larger and bold title
plt.xticks(index, labels, fontsize=10)  # Adjust font size of x-axis ticks
plt.yticks(fontsize=10)  # Adjust font size of y-axis ticks
plt.legend(fontsize=10)  # Adjust font size of legend

plt.show()