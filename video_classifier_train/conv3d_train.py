from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, MaxPool3D, Dropout
from keras.layers.convolutional import Conv3D
from keras.utils import np_utils
import numpy as np
from video_classifier_train.ucf.UCF101_loader import load_ucf
from video_classifier_train.ucf.UCF101_extractor import scan_and_extract_features, MAX_NB_CLASSES
from sklearn.model_selection import train_test_split

BATCH_SIZE = 64
EPOCHS = 20
VERBOSE = 1

def main():
    data_dir_path = './very_large_data'
    model_dir_path = './models/UCF-101'
    weight_file_path = model_dir_path + '/conv3d-weights.h5'
    architecture_file_path = model_dir_path + '/conv3d-architecture.json'
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
        x_samples[i] = x[0:min_frames, :, :, :] / 255
    for y in y_samples:
        if y not in labels:
            labels[y] = len(labels)
    for i in range(len(y_samples)):
        y_samples[i] = labels[y_samples[i]]

    y_samples = np_utils.to_categorical(y_samples, MAX_NB_CLASSES)

    model = Sequential()
    model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), input_shape=(min_frames, img_width, img_height, img_channel), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D(pool_size=2))
    model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D(pool_size=2))
    model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    open(architecture_file_path, 'w').write(model.to_json())
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(x_samples, y_samples, test_size=0.2, random_state=42)

    checkpoint = ModelCheckpoint(filepath=weight_file_path, save_best_only=True)
    model.fit(x=Xtrain, y=Ytrain, batch_size=BATCH_SIZE,epochs=EPOCHS, verbose=VERBOSE,validation_data=(Xtest, Ytest),
              callbacks=[checkpoint])
    model.save_weights(weight_file_path)


if __name__ == '__main__':
    main()