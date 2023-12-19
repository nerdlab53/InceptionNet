from Inception import inception_module
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import datasets
import keras

def GoogleNet():
  inp = Input(shape=(32, 32, 3))
  # resizing the images
  input_tensor = tf.keras.layers.experimental.preprocessing.Resizing(224, 224, 
  interpolation='bilinear', input_shape = ([32, 32, 3]))(inp)
  x = Conv2D(64, 7, strides=2, padding='same', activation='relu', name='convolution1')(input_tensor)
  x = MaxPooling2D(3, strides=2, name='max_pool1')(x)
  x = Conv2D(64, 1, strides=1, padding='same', activation='relu', name='convolution2a')(x)
  x = Conv2D(192, 3, strides=1, padding='same', activation='relu',name='convolution2b')(x)
  x = MaxPooling2D(3, strides=2, name='maxpool2')(x)
  x = inception_module(x, 64, 96, 128, 16, 32, 32, name='inception_module 3(a)')
  x = inception_module(x, 128, 128, 192, 32, 96, 64, name='inception_module 3(b)')
  x = MaxPooling2D(3, strides=2, name='maxpool3')(x)
  x = inception_module(x, 192, 96, 208, 16, 48, 64, name='inception_module 4(a)')
  x0 = AveragePooling2D((5,5), strides=3)(x)
  x0 = Conv2D(128, 1, padding='same', activation='relu')(x0)
  x0 = Flatten()(x0)
  x0 = Dense(1024, activation='relu')(x0)
  x0 = Dropout(0.7)(x0)
  x0 = Dense(10, activation='softmax', name='aux_out_1')(x0)
  x = inception_module(x, 160, 112, 224, 24, 64, 64, name='inception_module 4(b)')
  x = inception_module(x, 128, 128, 256, 24, 64, 64, name='inception_module 4(c)')
  x = inception_module(x, 112, 144, 288, 32, 64, 64, name='inception_module 4(d)')
  x1 = MaxPooling2D((5,5), strides=3)(x)
  x1 = Conv2D(128, 1, padding='same', activation='relu')(x1)
  x1 = Flatten()(x1)
  x1 = Dense(1024, activation='relu')(x1)
  x1 = Dropout(0.7)(x1)
  x1 = Dense(10, activation='softmax', name='aux_out_2')(x1)
  x = inception_module(x, 256, 160, 320, 32, 128, 128, name='inception_module 4(e)')
  x = MaxPooling2D(3, padding='same', strides=2)(x)
  x = inception_module(x, 256, 160, 320, 32, 128, 128, name='inception_module 5(a)')
  x = inception_module(x, 384, 192, 384, 48, 128, 128, name='inception_module 5(b)')
  x = GlobalAveragePooling2D(name='global_avg_pool')(x)
  x = Dropout(0.4)(x)
  x = Dense(4, activation='softmax', name='output')(x)
  model = Model(inp, [x, x0, x1], name='inception_module_v1')
  return model
