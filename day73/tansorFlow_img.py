import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
# Load the InceptionV3 model pre-trained on ImageNet data
model = InceptionV3(weights='imagenet')
# Load and preprocess the image
img_path = 'path_to_your_image.jpg'  # Replace with your image path
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
# Make predictions
predictions = model.predict(x)
decoded_predictions = decode_predictions(predictions, top=3)[0]
# Print the top 3 predictions
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i + 1}: {label} ({score:.2f})")
