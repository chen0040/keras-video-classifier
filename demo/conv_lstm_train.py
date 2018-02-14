""" This script demonstrates the use of a convolutional LSTM network.
This network is used to predict the next frame of an artificially
generated movie which contains moving squares.
"""
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization

from keras import backend as K

K.set_image_dim_ordering('tf')

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.utils import np_utils
import numpy as np
from keras_video_classifier.library.utility.ucf.UCF101_loader import load_ucf
from keras_video_classifier.library.utility import scan_and_extract_features, MAX_NB_CLASSES
from sklearn.model_selection import train_test_split

BATCH_SIZE = 8
NUM_EPOCHS = 20
VERBOSE = 1


def generate_batch(x_samples, y_samples):
    num_batches = len(x_samples) // BATCH_SIZE

    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * BATCH_SIZE
            end = (batchIdx + 1) * BATCH_SIZE
            yield np.array(x_samples[start:end]), np.array(y_samples[start:end])


def main():
    data_dir_path = './very_large_data'
    model_dir_path = './models/UCF-101'
    weight_file_path = model_dir_path + '/conv3d-weights.h5'
    architecture_file_path = model_dir_path + '/conv3d-architecture.json'
    nb_classes = MAX_NB_CLASSES

    np.random.seed(42)

    max_frames = 0
    min_frames = 1000
    labels = dict()
    load_ucf(data_dir_path)
    x_samples, y_samples = scan_and_extract_features(data_dir_path)
    _, img_width, img_height, img_channel = x_samples[0].shape
    for x in x_samples:
        frames = x.shape[0]
        max_frames = max(frames, max_frames)
        min_frames = min(frames, min_frames)
    for i in range(len(x_samples)):
        x = x_samples[i]
        x = x[0:min_frames, :, :, :] / 255
        x_samples[i] = x
    for y in y_samples:
        if y not in labels:
            labels[y] = len(labels)
    for i in range(len(y_samples)):
        y_samples[i] = labels[y_samples[i]]

    y_samples = np_utils.to_categorical(y_samples, nb_classes)

    # We create a layer which take as input movies of shape
    # (n_frames, width, height, channels) and returns the categorical label

    model = Sequential()
    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                         input_shape=(None, img_width, img_height, img_channel),
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())

    model.add(Conv3D(filters=nb_classes, kernel_size=(3, 3, 3),
                     activation='softmax',
                     padding='same', data_format='channels_last'))

    # model.add(Dense(nb_classes))

    # model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    open(architecture_file_path, 'w').write(model.to_json())
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(x_samples, y_samples, test_size=0.2, random_state=42)

    train_gen = generate_batch(Xtrain, Ytrain)
    test_gen = generate_batch(Xtest, Ytest)

    train_num_batches = len(Xtrain) // BATCH_SIZE
    test_num_batches = len(Xtest) // BATCH_SIZE

    checkpoint = ModelCheckpoint(filepath=weight_file_path, save_best_only=True)
    model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                        epochs=NUM_EPOCHS,
                        verbose=1, validation_data=test_gen, validation_steps=test_num_batches, callbacks=[checkpoint])
    model.save_weights(weight_file_path)


if __name__ == '__main__':
    main()
