# keras-video-classifier-web-api

Keras implementation of video classifiers serving as web

The training data is [UCF101 - Action Recognition Data Set](http://crcv.ucf.edu/data/UCF101.php)

The opencv-python is used to extract frames from the videos.

# Deep Learning Models

The following deep learning models have been implemented and studied:

* VGG16+LSTM: this approach uses VGG16 to extract features from individual frame of the video, the sequence of frame features are then taken into LSTM recurrent networks for classifier.
    * training: video_classifier_train/vgg16_lstm_train.py 
    * predictor: video_classifier_web/vgg16_lstm_predict.py
    
* VGG16+Bidirectional LSTM: this approach uses VGG16 to extract features from individual frame of the video, the sequence of frame features are then taken into bidirectional LSTM recurrent networks for classifier.
    * training: video_classifier_train/vgg16_bidirectional_lstm_train.py 
    * predictor: video_classifier_web/vgg16_bidirectional_lstm_predict.py
    
# Evaluation

20 classes from UCF101 is used to train the video classifier. 20 epochs are set for the training

Below is the train history for the VGG16+LSTM:

![vgg16-lstm-history](/video_classifier_train/reports/vgg16-lstm-history.png)

Below is the train history for the VGG16+Bidirectional LSTM:

![vgg16-lstm-history](/video_classifier_train/reports/vgg16-bidirectional-lstm-history.png)
