from __future__ import print_function
import tensorflow as tf
from keras import backend as K

# Make sure that you have tensorflow-gpu installed if you want o use init_devices('gpu')


def init_devices(device_type=None):
    if device_type is None:
        device_type = 'cpu'

    num_cores = 4

    if device_type == 'gpu':
        num_GPU = 1
        num_CPU = 1
    else:
        num_CPU = 1
        num_GPU = 0

    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                            inter_op_parallelism_threads=num_cores, allow_soft_placement=True,
                            device_count={'CPU': num_CPU, 'GPU': num_GPU})
    session = tf.Session(config=config)
    K.set_session(session)


def print_devices():
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
