import numpy as np
from keras import backend as K

from video_classifier.utility.plot_utils import plot_and_save_history

from video_classifier.library.recurrent_networks import VGG16LSTMVideoClassifier
from video_classifier.utility.ucf.UCF101_loader import load_ucf

K.set_image_dim_ordering('tf')


def main():
    input_dir_path = './very_large_data'
    output_dir_path = './models/UCF-101'
    report_dir_path = './reports/UCF-101'

    np.random.seed(42)

    # this line downloads the video files of UCF-101 dataset if they are not available in the very_large_data folder
    load_ucf(input_dir_path)

    classifier = VGG16LSTMVideoClassifier()

    history = classifier.fit(data_dir_path=input_dir_path, model_dir_path=output_dir_path)

    plot_and_save_history(history, VGG16LSTMVideoClassifier.model_name,
                          report_dir_path + '/' + VGG16LSTMVideoClassifier.model_name + '-history.png')


if __name__ == '__main__':
    main()
