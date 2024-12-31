#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.model_selection import train_test_split
import json

import glob
import random
import collections
from glob import glob
import numpy as np
import pandas as pd

#import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from skimage.io import imread
from skimage.transform import resize
from keras import layers
from keras import models
import keras
from keras.layers import LeakyReLU
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, Activation, Flatten
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import shutil
import keras
from PIL import Image
#import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
import json
import tensorflow as tf 
from keras.layers import Input
from keras import Sequential
from keras.layers import Dense, LSTM,Flatten, TimeDistributed, Conv2D, Dropout
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import re
import string

from keras.models import load_model
from keras.callbacks import Callback,ModelCheckpoint
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import keras.backend as K
import numpy as np
import pandas as pd

from skimage.io import imread
from skimage.transform import resize
from keras import layers
from keras import models
import keras
from keras.layers import LeakyReLU
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Conv1D,Reshape, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, Activation, Flatten, UpSampling2D
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import pandas as pd

from skimage.io import imread
from skimage.transform import resize
from keras import layers
from keras import models
import keras
from keras.layers import LeakyReLU
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, Activation, Flatten
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.callbacks import Callback,ModelCheckpoint
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import keras.backend as K

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Lambda, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import Input
from keras.layers import TimeDistributed, Conv2D, Dense, MaxPooling2D, Flatten, LSTM, Dropout, BatchNormalization
from keras import models
AUTO = tf.data.AUTOTUNE
from sklearn.metrics import confusion_matrix
import random

AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 32
EPOCHS = 50
CROP_TO = 32
SEED = 26

PROJECT_DIM = 128
LATENT_DIM = 512
WEIGHT_DECAY = 0.0005
learning_rate = 0.0001
batch_size = 128
hidden_units = 512
projection_units = 256
num_epochs = 2
dropout_rate = 0.5

temperature = 0.05

# Define the path to the directory
directory_path = r'C:\Users\shaif\Downloads\Compressed\painting\painting'

# Initialize empty lists for images and labels
X_train = []
Y_train = []
class_label = 0

# Loop through subdirectories (classes)
for class_folder in tqdm(sorted(os.listdir(directory_path))):
    class_path = os.path.join(directory_path, class_folder)
    
    for image_file in os.listdir(class_path):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(class_path, image_file)
            
            # Load image, convert to RGB and resize
            img = Image.open(image_path).convert('RGB')
            img = img.resize((32, 32))
            img_array = np.array(img)
            
            # Append image and label to lists
            X_train.append(img_array)
            Y_train.append(class_label)
            
    class_label = class_label + 1

# Convert lists to numpy arrays
X_train = np.array(X_train)
Y_train = np.array(Y_train)

print(f'X_train shape: {X_train.shape}')
print(f'Y_train shape: {Y_train.shape}')


# In[ ]:





# In[2]:


import os
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# Define the path to the directory
directory_path = r'C:\Users\shaif\Downloads\Compressed\real\real'

# Initialize empty lists for images and labels
X_test = []
Y_test = []
class_label = 0
# Loop through subdirectories (classes)
for class_folder in tqdm(sorted(os.listdir(directory_path))):
    class_path = os.path.join(directory_path, class_folder)

    for image_file in os.listdir(class_path):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(class_path, image_file)
            
            # Load image, convert to RGB and resize
            img = Image.open(image_path).convert('RGB')
            img = img.resize((32, 32))
            img_array = np.array(img)
            
            # Append image and label to lists
            X_test.append(img_array)
            Y_test.append(class_label)
            
    class_label = class_label + 1

# Convert lists to numpy arrays
X_test = np.array(X_test)
Y_test = np.array(Y_test)

print(f'X_train shape: {X_test.shape}')
print(f'Y_train shape: {Y_test.shape}')


# In[3]:


from tensorflow.keras.utils import to_categorical
num_classes = 345
Y_train_categorical = to_categorical(Y_train, num_classes=num_classes)
Y_test_categorical = Y_test


# In[ ]:





# In[4]:


from sklearn.model_selection import train_test_split
x_encode, X_test, y_encode, Y_test = train_test_split(X_test, Y_test_categorical, test_size=0.2, random_state=42)


# In[7]:


data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.Normalization(),
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.02),
        layers.experimental.preprocessing.RandomWidth(0.2),
        layers.experimental.preprocessing.RandomHeight(0.2),
    ]
)

# Setting the state of the normalization layer.
data_augmentation.layers[0].adapt(x_encode)



num_classes = 345
input_shape = (32, 32, 3)

def create_encoder():
    resnet = tf.keras.applications.ResNet50V2( include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg" )

    inputs = keras.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    outputs = resnet(augmented)
    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10-encoder")
    return model

encoder = create_encoder()
encoder.summary()

def create_classifier(encoder, trainable=True):

    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    features = layers.Dropout(dropout_rate)(features)
    features = layers.Dense(hidden_units, activation="relu")(features)
    features = layers.Dropout(dropout_rate)(features)
    outputs = layers.Dense(num_classes, activation="softmax")(features)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10-classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):

        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)

        logits = tf.divide( tf.matmul(  
            feature_vectors_normalized, tf.transpose(feature_vectors_normalized)),self.temperature,)
        
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)


def add_projection_head(encoder):
    
    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    outputs = layers.Dense(projection_units, activation="relu")(features)
    model = keras.Model(
        inputs=inputs, outputs=outputs, name="cifar-encoder_with_projection-head"
    )
    
    return model

encoder = create_encoder()

encoder_with_projection_head = add_projection_head(encoder)

encoder_with_projection_head.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=SupervisedContrastiveLoss(temperature),
)

history = encoder_with_projection_head.fit(x=x_encode, y=y_encode, batch_size=256, epochs=20)
classifier = create_classifier(encoder, trainable=False) 
from keras.callbacks import EarlyStopping
# Define early stopping criteria
early_stopping_monitor = EarlyStopping(patience=10, monitor='val_loss', verbose=1)
# Train the model with early stopping
history = classifier.fit(X_train.astype('float32'), np.argmax(Y_train_categorical, axis=1), 
                         batch_size=32, epochs=80, 
                         callbacks=[early_stopping_monitor])




import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Assuming you have trained a model and obtained predicted probabilities on x_test
y_pred_prob = classifier.predict(X_test,batch_size= 64)

# Convert predicted probabilities to predicted labels
y_pred = np.argmax((y_pred_prob), axis=1)

# Convert y_test to predicted labels format
y_test_labels =  (Y_test)

# Calculate accuracy
accuracy = accuracy_score(y_test_labels, y_pred)
print("Accuracy:", accuracy)

# Calculate F1 score (micro-average)
f1_micro = f1_score(y_test_labels, y_pred, average='micro')
print("F1 Score (Micro):", f1_micro)








