#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[2]:


# Load the CIFAR-100 dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# Normalize the pixel values between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert the labels to one-hot encoding
y_train = keras.utils.to_categorical(y_train, 100)
y_test = keras.utils.to_categorical(y_test, 100)


# In[3]:


len(x_train)


# In[ ]:





# In[ ]:





# In[ ]:


#Method 1 


# In[2]:


import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
import glob

target_size = (32, 32)  # Change the values as per your requirement

# Load the pre-trained ResNet50 model with modified input shape
model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(target_size[0], target_size[1], 3))

# Define the path to the ImageNet dataset
dataset_path = 'D:/data/imagenet'

# Get the list of class folders in the dataset
class_folders = glob.glob(dataset_path+'/*/')


# In[ ]:





# In[231]:


import csv
import glob
import os
import numpy as np
import gc
from PIL import Image
from tqdm import tqdm

def get_features(image_path):
    
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
                    
    image = image.resize((32, 32))

    image = np.array(image).astype("float") / 255

    feature = model.predict((image).reshape(1,32,32,3))
    
    return feature



def extract_features_from_directory(directory, csv_file):
    
    total_features = []
    i = 0
    
    for image_path in tqdm(glob.glob(directory + '/*')):
        if (i>300):
            break
        try:
            f = get_features(image_path)
            total_features.append(f)
            i = i + 1
        except Exception as e:
            print(f"Error message: {str(e)}")
            continue
            
    average_features =  np.mean(total_features,axis = 0)
    
    
    with open(csv_file, 'a') as file:
        
        file.write(directory + '\t')
        
        for value in average_features[0]:
            file.write(str(value) + '\t')
        file.write('\n')
        
    return average_features


# In[ ]:





# In[ ]:





# In[239]:


from tqdm import tqdm

import csv

import csv

file_path = 'D:/feature.csv'
header = ["Name"] + ['Feature {}'.format(i) for i in range(2048)]  # Example header

with open(file_path, 'w') as file:
    
    for value in header:
        file.write(str(value) + '\t')
    file.write('\n')
    


z = None
# Iterate over all directories in class_folders
for directory in tqdm(class_folders, desc='Processing directories'):
    # Call the function to extract features and store them in the CSV file
    z = extract_features_from_directory(directory, file_path)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[80]:





# In[ ]:




