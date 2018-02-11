import cv2
import os
import numpy as np

MAX_NB_CLASSES = 4


def extract_images(video_input_file_path, image_output_dir_path):
    if os.path.exists(image_output_dir_path):
        return
    count = 0
    print('Extracting frames from video: ', video_input_file_path)
    vidcap = cv2.VideoCapture(video_input_file_path)
    success, image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        if success:
            cv2.imwrite(image_output_dir_path + os.path.sep + "frame%d.jpg" % count, image)  # save frame as JPEG file
            count = count + 1


def extract_features(video_input_file_path, feature_output_file_path):
    if os.path.exists(feature_output_file_path):
        return np.load(feature_output_file_path)
    count = 0
    print('Extracting frames from video: ', video_input_file_path)
    vidcap = cv2.VideoCapture(video_input_file_path)
    success, image = vidcap.read()
    features = []
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        if success:
            img = cv2.resize(image, (40, 40), interpolation=cv2.INTER_AREA)
            features.append(image)
            count = count + 1
    unscaled_features = np.array(features)
    print(unscaled_features.shape)
    np.save(feature_output_file_path, unscaled_features)
    return unscaled_features


def extract_videos_for_conv2d(video_input_file_path, feature_output_file_path, max_frames):
    if feature_output_file_path is not None:
        if os.path.exists(feature_output_file_path):
            return np.load(feature_output_file_path)
    count = 0
    print('Extracting frames from video: ', video_input_file_path)
    vidcap = cv2.VideoCapture(video_input_file_path)
    success, image = vidcap.read()
    features = []
    success = True
    while success and count < max_frames:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        if success:
            image = cv2.resize(image, (240, 240), interpolation=cv2.INTER_AREA)
            channels = image.shape[2]
            for channel in range(channels):
                features.append(image[:, :, channel])
            count = count + 1
    unscaled_features = np.array(features)
    unscaled_features = np.transpose(unscaled_features, axes=(1, 2, 0))
    print(unscaled_features.shape)
    if feature_output_file_path is not None:
        np.save(feature_output_file_path, unscaled_features)
    return unscaled_features


def scan_and_extract_images(data_dir_path):
    input_data_dir_path = data_dir_path + '/UCF-101'
    output_frame_data_dir_path = data_dir_path + '/UCF-101-Frames'
    
    if not os.path.exists(output_frame_data_dir_path):
        os.makedirs(output_frame_data_dir_path)

    dir_count = 0
    for f in os.listdir(input_data_dir_path):
        file_path = input_data_dir_path + os.path.sep + f
        if not os.path.isfile(file_path):
            output_dir_name = f
            output_dir_path = output_frame_data_dir_path + os.path.sep + output_dir_name
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)
            dir_count += 1
            for ff in os.listdir(file_path):
                video_file_path = file_path + os.path.sep + ff
                output_image_folder_path = output_dir_path + os.path.sep + ff.split('.')[0]
                if not os.path.exists(output_image_folder_path):
                    os.makedirs(output_image_folder_path)
                extract_images(video_file_path, output_image_folder_path)
        if dir_count == MAX_NB_CLASSES:
            break


def scan_and_extract_features(data_dir_path, data_set_name=None):
    if data_set_name is None:
        data_set_name = 'UCF-101'
    input_data_dir_path = data_dir_path + '/' + data_set_name
    output_feature_data_dir_path = data_dir_path + '/' + data_set_name + '-Features'
    
    if not os.path.exists(output_feature_data_dir_path):
        os.makedirs(output_feature_data_dir_path)

    y_samples = []
    x_samples = []

    dir_count = 0
    for f in os.listdir(input_data_dir_path):
        file_path = input_data_dir_path + os.path.sep + f
        if not os.path.isfile(file_path):
            output_dir_name = f
            output_dir_path = output_feature_data_dir_path + os.path.sep + output_dir_name
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)
            dir_count += 1
            for ff in os.listdir(file_path):
                video_file_path = file_path + os.path.sep + ff
                output_feature_file_path = output_dir_path + os.path.sep + ff.split('.')[0] + '.npy'
                x = extract_features(video_file_path, output_feature_file_path)
                y = f
                y_samples.append(y)
                x_samples.append(x)

        if dir_count == MAX_NB_CLASSES:
            break

    return x_samples, y_samples


def scan_and_extract_videos_for_conv2d(data_dir_path, data_set_name=None, max_frames=None):
    if data_set_name is None:
        data_set_name = 'UCF-101'
    if max_frames is None:
        max_frames = 10

    input_data_dir_path = data_dir_path + '/' + data_set_name
    output_feature_data_dir_path = data_dir_path + '/' + data_set_name + '-Conv2d'

    if not os.path.exists(output_feature_data_dir_path):
        os.makedirs(output_feature_data_dir_path)

    y_samples = []
    x_samples = []

    dir_count = 0
    for f in os.listdir(input_data_dir_path):
        file_path = input_data_dir_path + os.path.sep + f
        if not os.path.isfile(file_path):
            output_dir_name = f
            output_dir_path = output_feature_data_dir_path + os.path.sep + output_dir_name
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)
            dir_count += 1
            for ff in os.listdir(file_path):
                video_file_path = file_path + os.path.sep + ff
                output_feature_file_path = output_dir_path + os.path.sep + ff.split('.')[0] + '.npy'
                x = extract_videos_for_conv2d(video_file_path, output_feature_file_path, max_frames)
                y = f
                y_samples.append(y)
                x_samples.append(x)

        if dir_count == MAX_NB_CLASSES:
            break

    return x_samples, y_samples


def main():
    print(cv2.__version__)
    data_dir_path = '.././very_large_data'
    X, Y = scan_and_extract_videos_for_conv2d(data_dir_path)
    print(X[0].shape)


if __name__ == '__main__':
    main()
