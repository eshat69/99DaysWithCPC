import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
sns.set_style('darkgrid')
import shutil
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import time
from tqdm import tqdm
from sklearn.metrics import f1_score
from IPython.display import YouTubeVideo
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199
print ('Modules loaded')

import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def make_dataframes(sdir):
    bad_images = []
    classes = ['cheetah', 'fox', 'hyena', 'lion', 'tiger', 'wolf']
    filepaths = []
    labels = []

    # Iterate through each class directory
    for klass in sorted(os.listdir(sdir)):
        classpath = os.path.join(sdir, klass)
        subdir = os.listdir(classpath)[0]
        subpath = os.path.join(classpath, subdir)

        if os.path.isdir(subpath):
            flist = sorted(os.listdir(subpath))
            for f in tqdm(flist, desc=f'Processing {klass}', unit='files', colour='blue'):  # Corrected 'colour' to 'bar_format'
                fpath = os.path.join(subpath, f)
                try:
                    img = cv2.imread(fpath)
                    if img is not None:
                        filepaths.append(fpath)
                        labels.append(klass)
                    else:
                        bad_images.append(fpath)
                except Exception as e:
                    bad_images.append(fpath)

    # Create DataFrame
    df = pd.DataFrame({'filepaths': filepaths, 'labels': labels})

    # Split into train, validation, and test sets
    train_df, dummy_df = train_test_split(df, train_size=0.7, shuffle=True, random_state=123, stratify=df['labels'])
    valid_df, test_df = train_test_split(dummy_df, train_size=0.5, shuffle=True, random_state=123, stratify=dummy_df['labels'])

    # Print some statistics
    class_count = len(classes)
    print(f'Number of classes in processed dataset: {class_count}')
    print(f'Train_df length: {len(train_df)}, Test_df length: {len(test_df)}, Valid_df length: {len(valid_df)}')

    # Calculate average image dimensions
    sample_df = train_df.sample(n=50, replace=False)
    ht, wt, count = 0, 0, 0
    for fpath in sample_df['filepaths']:
        img = cv2.imread(fpath)
        if img is not None:
            h, w = img.shape[:2]
            ht += h
            wt += w
            count += 1
    ave_h = int(ht / count)
    ave_w = int(wt / count)
    aspect_ratio = ave_h / ave_w
    print(f'Average image height: {ave_h}, width: {ave_w}, aspect ratio (h/w): {aspect_ratio}')

    # Print bad images if any
    if bad_images:
        print('Invalid image files:')
        for f in bad_images:
            print(f)

    return train_df, test_df, valid_df, classes, class_count

# Example usage
sdir = r'/content/drive/MyDrive/wild animals'
train_df, test_df, valid_df, classes, class_count = make_dataframes(sdir)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def make_gens(batch_size, train_df, test_df, valid_df, img_size):
    # Create ImageDataGenerator instances
    train_datagen = ImageDataGenerator()
    test_valid_datagen = ImageDataGenerator()

    # Create training generator
    print("Creating training generator...", end=' ')
    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filepaths',
        y_col='labels',
        target_size=img_size,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True,
        batch_size=batch_size
    )
    print("Done.")

    # Create validation generator
    print("Creating validation generator...", end=' ')
    valid_gen = test_valid_datagen.flow_from_dataframe(
        dataframe=valid_df,
        x_col='filepaths',
        y_col='labels',
        target_size=img_size,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False,
        batch_size=batch_size  # Use the same batch size as training for consistency
    )
    print("Done.")

    # Create test generator
    print("Creating test generator...", end=' ')
    test_batch_size = find_optimal_batch_size(len(test_df), max_batch_size=batch_size)  # Use specified max_batch_size
    test_steps = len(test_df) // test_batch_size
    test_gen = test_valid_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filepaths',
        y_col='labels',
        target_size=img_size,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False,
        batch_size=test_batch_size
    )
    print("Done.")

    # Extract class information
    classes = list(train_gen.class_indices.keys())
    class_count = len(classes)

    # Print test generator details
    print(f"Test batch size: {test_batch_size}, Test steps: {test_steps}, Number of classes: {class_count}")

    return train_gen, test_gen, valid_gen, test_steps

def find_optimal_batch_size(dataset_length, max_batch_size=80):
    """
    Finds the optimal batch size for the test generator such that:
    - The batch size divides the dataset length evenly.
    - The batch size is as large as possible but <= max_batch_size.
    """
    for batch_size in range(max_batch_size, 0, -1):
        if dataset_length % batch_size == 0:
            return batch_size
    return 1  # Fallback to 1 if no suitable batch size is found

# Example usage
batch_size = 20
img_size = (224, 224)  # Example image size

train_gen, test_gen, valid_gen, test_steps = make_gens(batch_size, train_df, test_df, valid_df, img_size)

import matplotlib.pyplot as plt
import numpy as np

def show_image_samples(gen):
    """
    Displays a sample of images from a generator.

    Parameters:
        gen: A generator created using `ImageDataGenerator.flow_from_dataframe`.
    """
    # Get class indices and class names
    class_indices = gen.class_indices
    classes = list(class_indices.keys())

    # Get a batch of images and labels from the generator
    images, labels = next(gen)

    # Normalize images to [0, 1] range (if not already normalized)
    images = images / 255.0

    # Determine the number of images to display (maximum of 25)
    num_images = min(len(labels), 25)

    # Set up the plot
    plt.figure(figsize=(15, 15))

    # Display each image
    for i in range(num_images):
        plt.subplot(5, 5, i + 1)  # Arrange images in a 5x5 grid
        plt.imshow(images[i])

        # Get the class name for the image
        index = np.argmax(labels[i])
        class_name = classes[index]

        # Add the class name as the title
        plt.title(class_name, color='blue', fontsize=12)
        plt.axis('off')  # Hide axes

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

# Example usage
show_image_samples(train_gen)

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import regularizers

def make_model(img_size, lr, mod_num=50, class_count=6):
    img_shape = (img_size[0], img_size[1], 3)

    if mod_num == 50:
        base_model = ResNet50(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')
        msg = 'Created ResNet-50 model'
    else:
        base_model = ResNet50(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')
        msg = 'Created ResNet-50 model (default)'

    base_model.trainable = True

    x = base_model.output
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = Dense(256, kernel_regularizer=regularizers.l2(l2=0.016), # Changed 'l' to 'l2'
              activity_regularizer=regularizers.l1(0.006),
              bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
    x = Dropout(rate=0.4, seed=123)(x)
    output = Dense(class_count, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(Adamax(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    msg = msg + f' with initial learning rate set to {lr}'
    print(msg)

    return model

# Example usage with ResNet-50
img_size = (224, 224)  # Example image size
lr = 0.001
model = make_model(img_size, lr, mod_num=50)

class LR_ASK(tf.keras.callbacks.Callback):
    def __init__(self, model, epochs, ask_epoch, dwell=True, factor=0.4):
        super(LR_ASK, self).__init__()
        self.model = model
        self.ask_epoch = ask_epoch
        self.epochs = epochs
        self.ask = True
        self.lowest_vloss = np.inf
        self.lowest_aloss = np.inf
        self.best_weights = self.model.get_weights()
        self.best_epoch = 1
        self.plist = []
        self.alist = []
        self.dwell = dwell
        self.factor = factor

    def on_train_begin(self, logs=None):
        if self.ask_epoch == 0:
            print('You set ask_epoch = 0, ask_epoch will be set to 1.')
            self.ask_epoch = 1
        if self.ask_epoch >= self.epochs:
            print('ask_epoch >= epochs. Training will proceed for', self.epochs, 'epochs.')
            self.ask = False
        if self.epochs == 1:
            self.ask = False
        else:
            print(f'Training will proceed until epoch {self.ask_epoch}.')
            print('Enter H to halt training or enter an integer for how many more epochs to run.')
            if self.dwell:
                print('Learning rate will be automatically adjusted during training.')

    def on_train_end(self, logs=None):
        print(f'Loading model with weights from epoch {self.best_epoch}.')
        self.model.set_weights(self.best_weights)
        tr_duration = time.time() - self.start_time
        hours = tr_duration // 3600
        minutes = (tr_duration - (hours * 3600)) // 60
        seconds = tr_duration - ((hours * 3600) + (minutes * 60))
        print(f'Training elapsed time: {int(hours)} hours, {minutes:.1f} minutes, {seconds:.2f} seconds.')

    def on_epoch_end(self, epoch, logs=None):
        vloss = logs.get('val_loss')
        aloss = logs.get('loss')

        if epoch > 0:
            deltav = self.lowest_vloss - vloss
            pimprov = (deltav / self.lowest_vloss) * 100
            self.plist.append(pimprov)

            deltaa = self.lowest_aloss - aloss
            aimprov = (deltaa / self.lowest_aloss) * 100
            self.alist.append(aimprov)
        else:
            pimprov = 0.0
            aimprov = 0.0

        if vloss < self.lowest_vloss:
            self.lowest_vloss = vloss
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch + 1
            print(f'\nValidation loss of {vloss:.4f} is {pimprov:.4f}% below lowest loss. Saving weights from epoch {epoch + 1}.')
        else:
            pimprov = abs(pimprov)
            print(f'\nValidation loss of {vloss:.4f} is {pimprov:.4f}% above lowest loss of {self.lowest_vloss:.4f}. Keeping weights from epoch {self.best_epoch}.')

            if self.dwell:
                lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                new_lr = lr * self.factor
                print(f'Learning rate adjusted from {lr:.6f} to {new_lr:.6f}. Model weights set to best weights.')
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                self.model.set_weights(self.best_weights)

        if aloss < self.lowest_aloss:
            self.lowest_aloss = aloss

        if self.ask:
            if epoch + 1 == self.ask_epoch:
                print('\nEnter H to end training or enter an integer for the number of additional epochs to run.')
                ans = input()

                if ans.lower() == 'h' or ans == '0':
                    print(f'You entered {ans}. Training halted on epoch {epoch + 1} due to user input.\n')
                    self.model.stop_training = True
                else:
                    self.ask_epoch += int(ans)
                    if self.ask_epoch > self.epochs:
                        print(f'You specified maximum epochs as {self.epochs}. Cannot train for {self.ask_epoch}.')
                    else:
                        print(f'You entered {ans}. Training will continue to epoch {self.ask_epoch}.')
                        if not self.dwell:
                            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                            print(f'Current learning rate is {lr:.6f}. Press enter to keep this LR or enter a new LR.')
                            ans = input()
                            if ans != '':
                                new_lr = float(ans)
                                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                                print(f'Changed LR to {new_lr}.')
                            else:
                                print(f'Keeping current LR of {lr:.5f}.')

    def get_list(self):
        return self.plist, self.alist

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Assuming you have train_df, valid_df, img_size, and batch_size defined
# Create the ImageDataGenerator instances
train_datagen = ImageDataGenerator()
valid_datagen = ImageDataGenerator()

# Assuming num_classes based on the number of unique labels
num_classes = len(train_df['labels'].unique())

# Create the data generators
train_data = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepaths',  # Replace with the actual column name for file paths
    y_col='labels',     # Replace with the actual column name for labels
    target_size=img_size,
    class_mode='categorical',  # Assuming categorical for multi-class classification
    batch_size=batch_size,
    shuffle=True  # Shuffle the training data
)

val_data = valid_datagen.flow_from_dataframe(
    dataframe=valid_df,
    x_col='filepaths',  # Replace with the actual column name for file paths
    y_col='labels',     # Replace with the actual column name for labels
    target_size=img_size,
    class_mode='categorical',  # Assuming categorical for multi-class classification
    batch_size=batch_size,
    shuffle=False  # Validation data shouldn't be shuffled
)

# Define training parameters
epochs = 40
ask_epoch = 10

# Create an instance of LR_ASK callback
lr_ask_callback = LR_ASK(model, epochs=epochs, ask_epoch=ask_epoch, dwell=True, factor=0.4)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
num_classes = 6
history = model.fit(
    train_data,
    epochs=epochs,
    callbacks=[lr_ask_callback],
    validation_data=val_data,
    verbose=1  # Adjust verbosity as needed (0 = silent, 1 = progress bar, 2 = one line per epoch)
)