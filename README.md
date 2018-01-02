# keras-video-classifier-web-api

Keras implementation of video classifiers serving as web

The training data is [UCF101 - Action Recognition Data Set](http://crcv.ucf.edu/data/UCF101.php). 
Codes are included that will download the UCF101 if they do not exist (due to their large size) in 
the video_classifier/training/very_large_data folder. The download utility codes can be found in
video_classifier/utility/ucf directory

The video classifiers are defined and implemented in the video_classifier/library directory. 

By default the classifiers are trained using video files inside the dataset "UCF-101" located in 
video_classifier/training/very_large_data (the videos files will be downloaded if not exist during
training). However, the classifiers are generic and can be used to train on any other datasets 
(just change the dataset_name parameter in its fit() method to other dataset name instead of UCF-101
will allow it to be trained on other video datasets)

The opencv-python is used to extract frames from the videos.

# Deep Learning Models

The following deep learning models have been implemented and studied:

* VGG16+LSTM: this approach uses VGG16 to extract features from individual frame of the video, the sequence of frame features are then taken into LSTM recurrent networks for classifier.
    * training: training/vgg16_lstm_train.py 
    * predictor: training/vgg16_lstm_predict.py
    * training: training/vgg16_lstm_hi_dim_train.py (VGG16 top not included) 
    * predictor: training/vgg16_lstm_hi_dim_predict.py (VGG16 top not included)
    
* VGG16+Bidirectional LSTM: this approach uses VGG16 to extract features from individual frame of the video, the sequence of frame features are then taken into bidirectional LSTM recurrent networks for classifier.
    * training: training/vgg16_bidirectional_lstm_train.py 
    * predictor: training/vgg16_bidirectional_lstm_predict.py
    * training: training/vgg16_bidirectional_lstm_hi_dim_train.py (VGG16 top not included)
    * predictor: training/vgg16_bidirectional_lstm_hi_dim_predict.py (VGG16 top not included)
    
The trained models are available in the video_classifier/training/models/UCF-101 folder 
(Weight files of two of the trained model are not included as they are too big to upload, they are 
* video_classifier/training/models/UCF-101/vgg16-lstm-hi-dim-weights.h5
* video_classifier/training/models/UCF-101/vgg16-bidirectional-lstm-hi-dim-weights.h5
)

# Usage

### Train Deep Learning model

To train a deep learning model, say VGG16BidirectionalLSTMVideoClassifier, run the following commands:

```bash
pip install requirements.txt

cd video_classifier/training
python vgg16_bidirectional_lstm_train.py 
```

The training code in vgg16_bidirectional_lstm_train.py is quite straightforward and illustrated below:

```python
import numpy as np
from keras import backend as K
from video_classifier.library.recurrent_networks import VGG16BidirectionalLSTMVideoClassifier
from video_classifier.utility.plot_utils import plot_and_save_history
from video_classifier.utility.ucf.UCF101_loader import load_ucf

K.set_image_dim_ordering('tf')

dataset_name = 'UCF-101'
input_dir_path = './very_large_data' # relative path to video_classifier/training/very_large_data
output_dir_path = './models/' + dataset_name # relative path to video_classifier/training/models/UCF-101
report_dir_path = './reports/' + dataset_name # relative path to video_classifier/training/reports/UCF-101

np.random.seed(42)

# this line downloads the video files of UCF-101 dataset if they are not available in the very_large_data folder
load_ucf(input_dir_path)

classifier = VGG16BidirectionalLSTMVideoClassifier()

history = classifier.fit(data_dir_path=input_dir_path, model_dir_path=output_dir_path, dataset_name=dataset_name)

plot_and_save_history(history, VGG16BidirectionalLSTMVideoClassifier.model_name,
                      report_dir_path + '/' + VGG16BidirectionalLSTMVideoClassifier.model_name + '-history.png')

```

After the training is completed, the trained models will be saved as cf-v1-*.* in the video_classifier/training/models.

### Predict Video Class Label

To use the trained deep learning model to predict the class label of a video, you can use the following code:

```python

import numpy as np

from video_classifier.library.recurrent_networks import VGG16BidirectionalLSTMVideoClassifier
from video_classifier.utility.ucf.UCF101_loader import load_ucf, scan_ucf

vgg16_include_top = True
dataset_name = 'UCF-101'
data_dir_path = '../training/very_large_data' # relative path to video_classifier/training/very_large_data
model_dir_path = '../training/models/' + dataset_name # relative path to video_classifier/training/models/UCF-101
config_file_path = VGG16BidirectionalLSTMVideoClassifier.get_config_file_path(model_dir_path,
                                                                              vgg16_include_top=vgg16_include_top)
weight_file_path = VGG16BidirectionalLSTMVideoClassifier.get_weight_file_path(model_dir_path,
                                                                              vgg16_include_top=vgg16_include_top)

np.random.seed(42)

# this line downloads the video files of UCF-101 dataset if they are not available in the very_large_data folder
load_ucf(data_dir_path)

predictor = VGG16BidirectionalLSTMVideoClassifier()
predictor.load_model(config_file_path, weight_file_path)

# scan_ucf returns a dictionary object of (video_file_path, video_class_label) where video_file_path
# is the key and video_class_label is the value
videos = scan_ucf(data_dir_path, predictor.nb_classes)

video_file_path_list = np.array([file_path for file_path in videos.keys()])
np.random.shuffle(video_file_path_list)

for video_file_path in video_file_path_list:
    label = videos[video_file_path]
    predicted_label = predictor.predict(video_file_path)
    print('predicted: ' + predicted_label + ' actual: ' + label)
```

# Evaluation

20 classes from UCF101 is used to train the video classifier. 20 epochs are set for the training

### Evaluate VGG16+LSTM (top included for VGG16)

Below is the train history for the VGG16+LSTM (top included for VGG16):

![vgg16-lstm-history](/video_classifier/training/reports/UCF-101/vgg16-lstm-history.png)

The LSTM with VGG16 (top included)feature extractor: (accuracy around 68.9% for training and 55% for validation)

### Evaluate VGG16+Bidirectional LSTM (top included for VGG16):

Below is the train history for the VGG16+Bidirectional LSTM (top included for VGG16):

![vgg16-bidirectional-lstm-history](/video_classifier/training/reports/UCF-101/vgg16-bidirectional-lstm-history.png)

The bidirectional LSTM with VGG16 (top included) feature extractor: (accuracy around 89% for training and 77% for validation)

### Evaluate VGG16+LSTM (top not included for VGG16)

Below is the train history for the VGG16+LSTM (top not included for VGG16):

![vgg16-lstm-history](/video_classifier/training/reports/UCF-101/vgg16-lstm-hi-dim-history.png)

The LSTM with VGG16 (top not included)feature extractor: (accuracy around 100% for training and 98.83% for validation)

### Evaluate VGG16+Bidirectional LSTM (top not included for VGG16)

Below is the train history for the VGG16+LSTM (top not included for VGG16):

![vgg16-lstm-history](/video_classifier/training/reports/UCF-101/vgg16-bidirectional-lstm-hi-dim-history.png)

The LSTM with VGG16 (top not included)feature extractor: (accuracy around 100% for training and 98.57% for validation)
