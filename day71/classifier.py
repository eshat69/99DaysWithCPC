from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd

# Example DataFrame (replace with your actual DataFrame)
df = pd.DataFrame({
    'age': [22, 38, 26, 35],
    'fare': [7.25, 71.28, 8.05, 53.1],
    'steps': [5000, 6000, 7000, 5500]  # Assuming 'steps' is a column in your data
})
# Calculate the mean of 'steps' column
mean_steps = df['steps'].mean()
# Define feature columns
my_feature_columns = [
    tf.feature_column.numeric_column('age'),
    tf.feature_column.numeric_column('fare'),
    tf.feature_column.numeric_column('steps')
]
# Input function
def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle and repeat if in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)
# Define the classifier
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[30, 10],
    n_classes=3
)
# Example usage of input_fn
features = {'age': df['age'], 'fare': df['fare'], 'steps': df['steps'] - mean_steps}
labels = [0, 1, 0, 1]  # Example labels, replace with your actual labels
input_fn(features, labels)  # Use input_fn as needed in your training or evaluation process
