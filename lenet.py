# ------------------------------------------------------------------------------------------
#   COMP 3106 - Introduction to Artificial Intelligence
#   Term Project:   Smile Detection
#   Description:    Script to build LeNet model
# ------------------------------------------------------------------------------------------
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as Ks

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)

        # update the input shape
        if Ks.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)

        # CONV => ReLU => POOL
        model.add(Conv2D(20, (5, 5), padding='same', input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # CONV => ReLU => POOL
        model.add(Conv2D(50, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # FC => ReLU
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))

        # softmax
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model