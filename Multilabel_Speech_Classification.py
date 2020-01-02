#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing General Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Importing the data

path = np.load('path.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
feat = np.load('feat.npy', mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Crating dataframes to match features, labels and paths

data = pd.DataFrame({'path': path, 'feat': feat})
data_complete = pd.merge(train, data, on = 'path')

# Transforming the columns in arrays

labels = np.array(data_complete['word'])
features = np.array(data_complete['feat'])

# Defining a padding function to padd the data

def padding(array):
    pad =[]
    for i in array:
        diff= 99 - i.shape[0]
        if diff ==0:
            pad.append(i)
        else:
            a= np.pad(i,((0,diff),(0,0)), mode='constant', constant_values=0)
            pad.append(a)
    return pad

padded_data = padding(features)
padded_data = np.array(padded_data)

# Flattening the array 

def flattening(array):
    lista = []
    for i in array:
        lista.append(i.flatten(order = 'C'))
    return np.array(lista)

train_data = flattening(padded_data)

# Splitting data in Train and Validation

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train_data, labels, test_size = 0.2, random_state = 42)
X_train.shape, X_val.shape, y_train.shape, y_val.shape

# Scaling the data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)

#Oversampling

from sklearn.utils import class_weight

class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

# Reshaping the data

X_train = X_train.reshape(75859,99,13)
X_val = X_val.reshape(18965,99,13)

# Encoding the labels

from sklearn.preprocessing import LabelBinarizer

onehot = LabelBinarizer()
y_train = onehot.fit_transform(y_train)
y_val   = onehot.transform(y_val)

# Importing the libraries to create our ANN

from keras import models
from keras import layers
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping

# MLP

# model = Sequential()
# model.add(layers.Dense(64, activation = 'relu', input_dim = X_train.shape[1]))
# model.add(layers.Dense(64, activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(layers.Dense(64, activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(layers.Dense(64, activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(layers.Dense(y_train.shape[1], activation = 'softmax'))

#model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# # CNN

# model = Sequential()
# model.add(Conv2D(128, kernel_size=(2, 2), activation='relu', input_shape=(99, 13, 1)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(128, kernel_size=(2, 2), activation = "relu"))
# model.add(Conv2D(64, kernel_size=(2, 2), activation = "relu"))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(35, activation='softmax'))

# model.compile(loss = 'categorical_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

# RNN 

import logging
logging.getLogger('tensorflow').disabled = True # Remove some unwanted warnings

model = Sequential()

model.add(LSTM(units=128, recurrent_dropout=0.35, return_sequences=True, input_shape = [99,13]))
model.add(Dropout(0.25))
model.add(LSTM(units=64, recurrent_dropout=0.35, return_sequences=False))
model.add(Dropout(0.25))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(units= y_train.shape[1], activation="softmax"))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Early_Stopping

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs= 20, batch_size = 100, callbacks=[es],
                    class_weight = class_weight, verbose = 0)

# Creating test dataset

data_test = pd.merge(test, data, on = 'path')
feature_test = np.array(data_test['feat'])
feature_test_padded = padding(feature_test)
feature_test_padded = np.array(feature_test_padded)  # shape (11005,99,13)
feature_test_padded= flattening(feature_test_padded) # shape (11005, 1287)
X_test = scaler.fit_transform(feature_test_padded)
X_test = X_test.reshape(11005,99,13)

# Prediction on test_set

pred_test = model.predict(X_test, verbose = 0)
pred_test = onehot.inverse_transform(pred_test)

# Create dataframe with results

results = pd.DataFrame(pred_test, columns = ["word"])
result = pd.concat([test, results], axis = 1)

# Saving the results in a csv file

result.to_csv("result.csv", index = False)

