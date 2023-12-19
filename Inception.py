import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import datasets
import keras


def inception_module(x, filters1x1, filters3x3_reduce, 
                    filters3x3, filters5x5_reduce, filters5x5, filters_pool, name=None):
        path1 = Conv2D(filters1x1, (1,1), padding='same', activation='relu')(x)
        path2 = Conv2D(filters3x3_reduce, (1, 1), padding='same', activation='relu')(x)
        path2 = Conv2D(filters3x3, (1, 1), padding='same', activation='relu')(path2)
        path3 = Conv2D(filters5x5_reduce, (1, 1), padding='same', activation='relu')(x)
        path3 = Conv2D(filters5x5, (1, 1), padding='same', activation='relu')(path3)
        path4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
        path4 = Conv2D(filters_pool, (1, 1), padding='same', activation='relu')(path4)
        return tf.concat([path1, path2, path3, path4], axis=3)