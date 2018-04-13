import numpy as np
from keras import backend as K
import os
from keras_video_classifier.library.utility.plot_utils import plot_and_save_history

from keras_video_classifier.library.convolutional import CnnVideoClassifier
from keras_video_classifier.library.utility.ucf.UCF101_loader import load_ucf

K.set_image_dim_ordering('tf')


def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)


def main():
    data_set_name = 'UCF-101'
    input_dir_path = patch_path('very_large_data')
    output_dir_path = patch_path('models/' + data_set_name)
    report_dir_path = patch_path('reports/' + data_set_name)

    np.random.seed(42)

    # this line downloads the video files of UCF-101 dataset if they are not available in the very_large_data folder
    load_ucf(input_dir_path)

    classifier = CnnVideoClassifier()

    history = classifier.fit(data_dir_path=input_dir_path, model_dir_path=output_dir_path,
                             data_set_name=data_set_name,
                             max_frames=10)

    plot_and_save_history(history, CnnVideoClassifier.model_name,
                          report_dir_path + '/' + CnnVideoClassifier.model_name + '-history.png')


if __name__ == '__main__':
    main()
