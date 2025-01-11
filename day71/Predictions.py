import tensorflow as tf
import numpy as np
# Define expected classes
expected = ['Setosa', 'Versicolor', 'Virginica']
# Define input data for prediction
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}
# Define input function to convert inputs to a Dataset
def input_fn(features, batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)
# Feature columns
feature_columns = []
for key in predict_x.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))
# Build the model
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 10],  # Example: two layers of 10 nodes each
    n_classes=len(expected),
    model_dir='./model'  # Directory where model checkpoints will be saved
)
# Prediction
predictions = list(classifier.predict(input_fn=lambda: input_fn(predict_x)))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    print('Predicted class is "{}" with probability {:.1f}%'.format(
        expected[class_id], 100 * probability))
