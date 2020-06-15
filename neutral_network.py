#Dogs and cats classification

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import zipfile
from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

!mkdir images
!wget --no-check-certificate https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip -O ./images/cats_and_dogs_filtered.zip

local_zip = 'images/cats_and_dogs_filtered.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('./images/')
zip_ref.close()

os.listdir("./images/cats_and_dogs_filtered")

width = 100
heigh = 100
batch_size = 32
num_classes = 2
epochs = 20

if K.image_data_format() == 'channels_first':
    input_shape = (3, width, heigh)
else:
    input_shape = (width, heigh, 3)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    validation_split=0.1,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '/content/images/cats_and_dogs_filtered/train',
    target_size=(width, heigh),
    batch_size=32,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    '/content/images/cats_and_dogs_filtered/validation',
    target_size=(width, heigh),
    batch_size=32,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    '/content/images/cats_and_dogs_filtered/train',
    target_size=(width, heigh),
    batch_size=32,
    class_mode='categorical'
)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.RMSprop(),
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
print(model.summary())

model.fit(train_generator, steps_per_epoch=train_generator.n//batch_size,
                    epochs=epochs, verbose=1, validation_data=validation_generator,
                    validation_steps=validation_generator.n//batch_size,
                    batch_size=32, callbacks=[early_stopping])

loss, acc = model.evaluate(test_generator, steps=200, verbose=1)
print(loss, acc)