from keras.layers import TimeDistributed, Input, Flatten, Dense
from keras.applications.inception_v3 import InceptionV3
from keras.layers.convolutional import Conv3D
from keras.models import Model

NB_CLASSES = 1000
NB_FILTERS = 40
IMG_WIDTH = 40
IMG_HEIGHT = 40
IMG_CHANNELS = 1
IMG_FRAMES = 1000

img_input = Input(shape=(IMG_FRAMES, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
base_cnn_model = InceptionV3(include_top=False)
temporal_analysis = TimeDistributed(base_cnn_model)(img_input)
conv3d_analysis = Conv3D(NB_FILTERS, 3, 3, 3)(temporal_analysis)
conv3d_analysis = Conv3D(NB_FILTERS, 3, 3, 3)(conv3d_analysis)
output = Flatten()(conv3d_analysis)
output = Dense(NB_CLASSES, activation="softmax")(output)

model = Model(img_input, output)

model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
