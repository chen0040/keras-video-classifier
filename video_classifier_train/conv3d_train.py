from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation
from keras.layers.convolutional import Conv3D
from keras.utils import np_utils
import numpy as np
from video_classifier_train.ucf.UCF101_loader import load_ucf
from video_classifier_train.ucf.UCF101_extractor import scan_and_extract_features


def main():
    data_dir_path = './very_large_data'
    feature_data_dir_path = data_dir_path + '/UCF-101-Features'

    max_frames = 0
    load_ucf(data_dir_path)
    x_samples, y_samples = scan_and_extract_features(data_dir_path)
    for x in x_samples:
        frames = x.shape[0]
        max_frames = max(frames, max_frames)
    for i in range(x_samples):
        first_frame = x[0, :, :, :]
        print(first_frame.shape)


if __name__ == '__main__':
    main()