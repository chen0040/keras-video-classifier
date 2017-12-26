""" This script demonstrates the use of a convolutional LSTM network.
This network is used to predict the next frame of an artificially
generated movie which contains moving squares.
"""
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Dropout, Bidirectional
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from video_classifier_train.ucf.UCF101_loader import load_ucf
from video_classifier_train.ucf.UCF101_vgg16_feature_extractor import scan_and_extract_vgg16_features, MAX_NB_CLASSES
from video_classifier_train.utils import plot_and_save_history

BATCH_SIZE = 64
NUM_EPOCHS = 20
VERBOSE = 1
HIDDEN_UNITS = 512
MAX_ALLOWED_FRAMES = 20

K.set_image_dim_ordering('tf')


def generate_batch(x_samples, y_samples):
    num_batches = len(x_samples) // BATCH_SIZE

    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * BATCH_SIZE
            end = (batchIdx + 1) * BATCH_SIZE
            yield np.array(x_samples[start:end]), y_samples[start:end]


def main():
    data_dir_path = './very_large_data'
    model_dir_path = './models/UCF-101'
    report_dir_path = './reports/UCF-101'
    model_name = 'vgg16-bidirectional-lstm'
    config_file_path = model_dir_path + '/' + model_name + '-config.npy'
    weight_file_path = model_dir_path + '/' + model_name + '-weights.h5'
    architecture_file_path = model_dir_path + '/' + model_name + '-architecture.json'
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
    for i in range(len(y_samples)):
        y_samples[i] = labels[y_samples[i]]

    y_samples = np_utils.to_categorical(y_samples, nb_classes)

    config = dict()
    config['labels'] = labels
    config['nb_classes'] = nb_classes
    config['num_input_tokens'] = num_input_tokens
    config['expected_frames'] = expected_frames

    np.save(config_file_path, config)

    model = Sequential()

    model.add(Bidirectional(LSTM(units=HIDDEN_UNITS, dropout=0.2, input_shape=(expected_frames, num_input_tokens))))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))

    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    open(architecture_file_path, 'w').write(model.to_json())
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(x_samples, y_samples, test_size=0.3, random_state=42)

    train_gen = generate_batch(Xtrain, Ytrain)
    test_gen = generate_batch(Xtest, Ytest)

    train_num_batches = len(Xtrain) // BATCH_SIZE
    test_num_batches = len(Xtest) // BATCH_SIZE

    checkpoint = ModelCheckpoint(filepath=weight_file_path, save_best_only=True)
    history = model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                  epochs=NUM_EPOCHS,
                                  verbose=1, validation_data=test_gen, validation_steps=test_num_batches,
                                  callbacks=[checkpoint])
    model.save_weights(weight_file_path)

    plot_and_save_history(history, model_name, report_dir_path + '/' + model_name + '-history.png')


if __name__ == '__main__':
    main()
