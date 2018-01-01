# keras-video-classifier-web-api

Keras implementation of video classifiers serving as web

The training data is [UCF101 - Action Recognition Data Set](http://crcv.ucf.edu/data/UCF101.php)

The opencv-python is used to extract frames from the videos.

# Deep Learning Models

The following deep learning models have been implemented and studied:

* VGG16+LSTM: this approach uses VGG16 to extract features from individual frame of the video, the sequence of frame features are then taken into LSTM recurrent networks for classifier.
    * training: video_classifier_train/vgg16_lstm_train.py 
    * predictor: video_classifier_train/vgg16_lstm_predict.py
    * training: video_classifier_train/vgg16_lstm_hi_dim_train.py (VGG16 top not included) 
    * predictor: video_classifier_train/vgg16_lstm_hi_dim_predict.py (VGG16 top not included)
    
* VGG16+Bidirectional LSTM: this approach uses VGG16 to extract features from individual frame of the video, the sequence of frame features are then taken into bidirectional LSTM recurrent networks for classifier.
    * training: video_classifier_train/vgg16_bidirectional_lstm_train.py 
    * predictor: video_classifier_train/vgg16_bidirectional_lstm_predict.py
    * training: video_classifier_train/vgg16_bidirectional_lstm_hi_dim_train.py (VGG16 top not included)
    * predictor: video_classifier_train/vgg16_bidirectional_lstm_hi_dim_predict.py (VGG16 top not included)
    
# Evaluation

20 classes from UCF101 is used to train the video classifier. 20 epochs are set for the training

### Evaluate VGG16+LSTM (top included for VGG16)

Below is the train history for the VGG16+LSTM (top included):

![vgg16-lstm-history](/video_classifier/video_classifier_train/reports/UCF-101/vgg16-lstm-history.png)

The LSTM with VGG16 (top included)feature extractor: (accuracy around 68.9% for training and 55% for validation)

### Evaluate VGG16+Bidirectional LSTM (top included for VGG16):

Below is the train history for the VGG16+Bidirectional LSTM:

![vgg16-bidirectional-lstm-history](/video_classifier/video_classifier_train/reports/UCF-101/vgg16-bidirectional-lstm-history.png)

The bidirectional LSTM with VGG16 (top included) feature extractor: (accuracy around 89% for training and 77% for validation)

### Evaluate VGG16+LSTM (top not included for VGG16)

Below is the train history for the VGG16+LSTM (top not included):

![vgg16-lstm-history](/video_classifier/video_classifier_train/reports/UCF-101/vgg16-lstm-hi-dim-history.png)

The LSTM with VGG16 (top not included)feature extractor: (accuracy around 100% for training and 98.83% for validation)
