# keras-video-classifier-web-api

Keras implementation of video classifiers serving as web

The following approaches have been implemented and compared:

* vgg16_lstm_train.py: this approach uses VGG16 to extract features from individual frame of the video, the sequence of frame features are then taken into LSTM recurrent networks for classifier 
