import numpy as np
from keras import backend as K
import os
import sys

K.set_image_dim_ordering('tf')


def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)


def main():
    sys.path.append(patch_path('..'))

    data_dir_path = patch_path('very_large_data')
    model_dir_path = patch_path('models/UCF-101')

    from keras_video_classifier.library.convolutional import CnnVideoClassifier
    from keras_video_classifier.library.utility.ucf.UCF101_loader import load_ucf, scan_ucf_with_labels
    config_file_path = CnnVideoClassifier.get_config_file_path(model_dir_path)
    weight_file_path = CnnVideoClassifier.get_weight_file_path(model_dir_path)

    np.random.seed(42)

    load_ucf(data_dir_path)

    predictor = CnnVideoClassifier()
    predictor.load_model(config_file_path, weight_file_path)

    videos = scan_ucf_with_labels(data_dir_path, [label for (label, label_index) in predictor.labels.items()])

    video_file_path_list = np.array([file_path for file_path in videos.keys()])
    np.random.shuffle(video_file_path_list)

    for video_file_path in video_file_path_list:
        label = videos[video_file_path]
        predicted_label = predictor.predict(video_file_path)
        print('predicted: ' + predicted_label + ' actual: ' + label)


if __name__ == '__main__':
    main()