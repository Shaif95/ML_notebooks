# imports

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.model_selection import train_test_split
import json
import glob
import random
from glob import glob
import matplotlib.pyplot as plt
import keras
from PIL import Image

# file dir

from glob import glob
trn1='test_data/invasive/*/'
trn2='test_data/noninvasive/*/'
tr1= glob(trn1)
tr2= glob(trn2)


# Load Model

from keras.models import load_model
model = load_model('base_model.h5')

# Image Data read

data = []
label = []
for i in tr1:
    for j in glob(i+'/*'):
        data.append(j)
        label.append(1)


for i in tr2:
    for j in glob(i+'/*'):
        data.append(j)
        label.append(0)
name = []
imgdata=[]
for i in range(len(data)):
    
    name.append(data[i])
    a = Image.open(data[i])
    b = a.resize((40, 40))
    c = np.array(b)
    imgdata.append(c.reshape(40,40,3))
    

from tensorflow.keras.utils import to_categorical
idata = np.array(imgdata)
X_test = idata
X_test = X_test.astype('float32') / 255.
X_test = np.reshape(X_test, (len(X_test),40,40,3))
# One hot vector representation of labels
Y_test = to_categorical(label)


# Predict

pred = model.predict(X_test)

# Create JSON

import json
p= pred.tolist()
# initialize the dictionary
my_dict = {}

# loop to emulate your data structure
for i in range(len(pred)):
    # assign a1 and a2
    a1 = i
    dict1 = {}
    dict1["name"] = name[i]
    dict1["label"] = label[i]
    x = np.round(p[i])
    if(x[0]== 0.0):
        y = 1
    else:
        y = 0
    dict1["pred"] = y
    a2 = p[i]
    
    
    dict1["pred_value"] = pred[i].tolist()
    
    
    # set a1 as the key and a2 as the value
    my_dict[a1] =  dict1

# use json.dump to write the file
with open('./square.json', 'w') as file:
    json.dump(my_dict, file, indent=4)





#F1 Function :


import keras.backend as K

def get_f1(y_true, y_pred): #taken from old keras source code
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    tn = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    
    fp = K.sum(K.round(K.clip((1-y_true) * (y_pred), 0, 1)))
    
    fn = K.sum(K.round(K.clip((y_true) * (1-y_pred), 0, 1)))
    

    f1_val = tp / ( tp + ( (1/2) * (fp+fn) ) + K.epsilon())
    
    return f1_val


# Cal predictions :

pred = model.predict(X_test)
p = np.round(pred)
f1 = get_f1(Y_test, p)


y_p = []
for i in range(len(p)):
    if ( p[i][0] == 0 ):
        y_p.append(1)
    else :
        y_p.append(0)
y_p = np.array(y_p)
y_t = []
for i in range(len(Y_test)):
    if ( Y_test[i][0] == 0 ):
        y_t.append(1)
    else :
        y_t.append(0)
y_t = np.array(y_t)

from sklearn.metrics import confusion_matrix
a=(confusion_matrix(y_t, y_p , labels=[0,1]))

from sklearn.metrics import accuracy_score , balanced_accuracy_score

ac = accuracy_score(Y_test, p)

# Print metrics :

import json
p= pred.tolist()
# initialize the dictionary
my_dict = {}


dict1 = {}
dict1["Accuracy"] = ac.tolist()
dict1["F1 Score"] = f1.numpy().tolist()
dict1["non_inv_true"] = a[0][0].tolist()
dict1["non_inv_false"] =  a[0][1].tolist()
dict1["inv_true"] = a[1][1].tolist()
dict1["inv_false"] = a[1][0].tolist()

my_dict["metrics"] =  dict1

# use json.dump to write the file
with open('./metrics.json', 'w') as file:
    json.dump(my_dict, file, indent=4)


