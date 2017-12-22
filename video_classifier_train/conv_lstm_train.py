""" This script demonstrates the use of a convolutional LSTM network.
This network is used to predict the next frame of an artificially
generated movie which contains moving squares.
"""
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np

NB_CLASSES = 100
IMG_WIDTH = 40
IMG_HEIGHT = 40
IMG_CHANNELS = 1

# We create a layer which take as input movies of shape
# (n_frames, width, height, channels) and returns the categorical label

seq = Sequential()
seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   input_shape=(None, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3)))

seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3)))

seq.add(Flatten())

seq.add(Dense(NB_CLASSES))

seq.add(Activation('softmax'))


seq.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

