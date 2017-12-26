""" This script demonstrates the use of a convolutional LSTM network.
This network is used to predict the next frame of an artificially
generated movie which contains moving squares.
"""
import numpy as np
from keras import backend as K
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD

from video_classifier_train.ucf.UCF101_loader import load_ucf, scan_ucf
from video_classifier_train.ucf.UCF101_vgg16_feature_extractor import extract_vgg16_features_live

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
    labels_idx2word = None
    model = None
    vgg16_model = None
    expected_frames = None

    def __init__(self):
        pass

    def load_model(self, config_file_path, weight_file_path):

        config = np.load(config_file_path).item()
        self.num_input_tokens = config['num_input_tokens']
        self.nb_classes = config['nb_classes']
        self.labels = config['labels']
        self.expected_frames = config['expected_frames']
        self.labels_idx2word = dict([(idx, word) for word, idx in self.labels.items()])

        model = Sequential()

        model.add(LSTM(units=HIDDEN_UNITS, input_shape=(None, self.num_input_tokens), return_sequences=False, dropout=0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(self.nb_classes))

        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        model.load_weights(weight_file_path)

        self.model = model

        vgg16_model = VGG16(include_top=True, weights='imagenet')
        vgg16_model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
        self.vgg16_model = vgg16_model

    def predict(self, video_file_path):
        x = extract_vgg16_features_live(self.vgg16_model, video_file_path)
        frames = x.shape[0]
        if frames > self.expected_frames:
            x = x[0:self.expected_frames, :]
        elif frames < self.expected_frames:
            temp = np.zeros(shape=(self.expected_frames, x.shape[1]))
            temp[0:frames, :] = x
            x = temp
        predicted_class = np.argmax(self.model.predict(np.array([x]))[0])
        predicted_label = self.labels_idx2word[predicted_class]
        return predicted_label


def main():
    model_name = 'vgg16-lstm'
    data_dir_path = '../video_classifier_train/very_large_data'
    model_dir_path = '../video_classifier_train/models/UCF-101'
    config_file_path = model_dir_path + '/' + model_name + '-config.npy'
    weight_file_path = model_dir_path + '/' + model_name + '-weights.h5'

    np.random.seed(42)

    load_ucf(data_dir_path)

    predictor = VGG16LSTMVideoClassifier()
    predictor.load_model(config_file_path, weight_file_path)

    videos = scan_ucf(data_dir_path, predictor.nb_classes)

    video_file_path_list = np.array([file_path for file_path in videos.keys()])
    np.random.shuffle(video_file_path_list)

    for video_file_path in video_file_path_list:
        label = videos[video_file_path]
        predicted_label = predictor.predict(video_file_path)
        print('predicted: ' + predicted_label + ' actual: ' + label)


if __name__ == '__main__':
    main()
