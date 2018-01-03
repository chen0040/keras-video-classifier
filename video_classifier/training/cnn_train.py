import numpy as np
from keras import backend as K

from video_classifier.utility.plot_utils import plot_and_save_history

from video_classifier.library.convolutional import CnnVideoClassifier
from video_classifier.utility.ucf.UCF101_loader import load_ucf

K.set_image_dim_ordering('tf')


def main():
    dataset_name = 'UCF-101'
    input_dir_path = './very_large_data'
    output_dir_path = './models/' + dataset_name
    report_dir_path = './reports/' + dataset_name

    np.random.seed(42)

    # this line downloads the video files of UCF-101 dataset if they are not available in the very_large_data folder
    load_ucf(input_dir_path)

    classifier = CnnVideoClassifier()

    history = classifier.fit(data_dir_path=input_dir_path, model_dir_path=output_dir_path, dataset_name=dataset_name, max_frames=10)

    plot_and_save_history(history, CnnVideoClassifier.model_name,
                          report_dir_path + '/' + CnnVideoClassifier.model_name + '-history.png')


if __name__ == '__main__':
    main()
