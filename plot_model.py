from Inception import inception_module
from GoogLeNet import GoogleNet
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import datasets
import keras

def main():
    model = GoogleNet()
    tf.keras.utils.plot_model(model, to_file='inception_alzheimer.png')

main()

if __name__ == "__main__":
    main()