import numpy as np

from video_classifier.library.recurrent_networks import VGG16BidirectionalLSTMVideoClassifier
from video_classifier.utility.ucf.UCF101_loader import load_ucf, scan_ucf


def main():

    vgg16_include_top = False
    data_dir_path = '../video_classifier_train/very_large_data'
    model_dir_path = '../video_classifier_train/models/UCF-101'
    config_file_path = VGG16BidirectionalLSTMVideoClassifier.get_config_file_path(model_dir_path,
                                                                                  vgg16_include_top=vgg16_include_top)
    weight_file_path = VGG16BidirectionalLSTMVideoClassifier.get_weight_file_path(model_dir_path,
                                                                                  vgg16_include_top=vgg16_include_top)

    np.random.seed(42)

    load_ucf(data_dir_path)

    predictor = VGG16BidirectionalLSTMVideoClassifier()
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
