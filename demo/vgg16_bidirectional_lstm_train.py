import numpy as np
from keras import backend as K
import os
import sys


def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)


def main():
    K.set_image_dim_ordering('tf')
    sys.path.append(patch_path('..'))

    from keras_video_classifier.library.recurrent_networks import VGG16BidirectionalLSTMVideoClassifier
    from keras_video_classifier.library.utility.plot_utils import plot_and_save_history
    from keras_video_classifier.library.utility.ucf.UCF101_loader import load_ucf

    data_set_name = 'UCF-101'
    input_dir_path = patch_path('very_large_data')
    output_dir_path = patch_path('models/' + data_set_name)
    report_dir_path = patch_path('reports/' + data_set_name)

    np.random.seed(42)

    # this line downloads the video files of UCF-101 dataset if they are not available in the very_large_data folder
    load_ucf(input_dir_path)

    classifier = VGG16BidirectionalLSTMVideoClassifier()

    history = classifier.fit(data_dir_path=input_dir_path, model_dir_path=output_dir_path, data_set_name=data_set_name)

    plot_and_save_history(history, VGG16BidirectionalLSTMVideoClassifier.model_name,
                          report_dir_path + '/' + VGG16BidirectionalLSTMVideoClassifier.model_name + '-history.png')


if __name__ == '__main__':
    main()
