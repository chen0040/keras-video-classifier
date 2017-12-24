""" This script demonstrates the use of a convolutional LSTM network.
This network is used to predict the next frame of an artificially
generated movie which contains moving squares.
"""
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from video_classifier_train.ucf.UCF101_loader import load_ucf
from video_classifier_train.ucf.UCF101_vgg16_feature_extractor import scan_and_extract_vgg16_features, MAX_NB_CLASSES

BATCH_SIZE = 64
NUM_EPOCHS = 20
VERBOSE = 1
HIDDEN_UNITS = 512
MAX_ALLOWED_FRAMES = 20

K.set_image_dim_ordering('tf')


class VGG16LSTMVideoClassifier(object):

    num_input_tokens = None
    nb_classes = None
    labels = None

    def __init__(self):
        pass

    def load_model(self, config_file_path, weight_file_path):

        config = np.load(config_file_path).items()
        self.num_input_tokens = config['num_input_tokens']
        self.nb_classes = config['nb_classes']
        self.labels = config['labels']

        model = Sequential()

        model.add(LSTM(units=HIDDEN_UNITS, input_shape=(None, self.num_input_tokens), return_sequences=False, dropout=0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(self.nb_classes))

        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        model.load_weights(weight_file_path)


def main():
    data_dir_path = '../video_classifier_train/very_large_data'
    model_dir_path = '../video_classifier_train/models/UCF-101'
    weight_file_path = model_dir_path + '/vgg16-lstm-weights.h5'
    nb_classes = MAX_NB_CLASSES

    np.random.seed(42)

    max_frames = 0
    labels = dict()
    load_ucf(data_dir_path)
    x_samples, y_samples = scan_and_extract_vgg16_features(data_dir_path)
    num_input_tokens = x_samples[0].shape[1]
    frames_list = []
    for x in x_samples:
        frames = x.shape[0]
        frames_list.append(frames)
        max_frames = max(frames, max_frames)
    expected_frames = int(np.mean(frames_list))
    print('max frames: ', max_frames)
    print('expected frames: ', expected_frames)
    for i in range(len(x_samples)):
        x = x_samples[i]
        frames = x.shape[0]
        if frames > expected_frames:
            x = x[0:expected_frames, :]
            x_samples[i] = x
        elif frames < expected_frames:
            temp = np.zeros(shape=(expected_frames, x.shape[1]))
            temp[0:frames, :] = x
            x_samples[i] = temp
    for y in y_samples:
        if y not in labels:
            labels[y] = len(labels)
    print(labels)
    labels_idx2word = dict([(idx, word) for word, idx in labels.items()])

    model = Sequential()

    model.add(LSTM(units=HIDDEN_UNITS, input_shape=(None, num_input_tokens), return_sequences=False, dropout=0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))

    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.load_weights(weight_file_path)

    for i in range(len(x_samples)):
        x = x_samples[i]
        predicted_class = np.argmax(model.predict(np.array([x]))[0])
        predicted_label = labels_idx2word[predicted_class]
        print('predicted: ' + predicted_label + ' actual: ' + y_samples[i])


if __name__ == '__main__':
    main()
