from imagesReader import *
import cv2
import random
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import Callback
from tensorflow.python.framework import ops
from keras.optimizers import Adam,SGD
from keras.models import load_model
ops.reset_default_graph()

if ((os.path.exists("vgg16_0[ImageGen-V2].h5"))):
    

    x_test, ImageName_ext = createTestdata()
    saved_model = load_model("vgg16_0[ImageGen-V2].h5")
    x = saved_model.predict(x_test)
    x = list(x)

    output2 = []
    for i in range(len(x)):
        output2.append([ImageName_ext[i],np.argmax(x[i])])

    createcsv(output2)
else:
    x_train, y_train = createTraindata()
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(shear_range=0.2, zoom_range=0.2, horizontal_flip=True,vertical_flip=True)

    model = Sequential()

    model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),name='vgg16'))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    ###############################################################
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.SGD(), metrics=['accuracy'])

    checkpoint = ModelCheckpoint("vgg16_0[ImageGen-V2].h5", monitor='val_accuracy', verbose=1, save_best_only=True)
    early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=30, verbose=1)


    model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=32), epochs= 256, verbose= 1,callbacks=[checkpoint, early],validation_data = (x_valid, y_valid))