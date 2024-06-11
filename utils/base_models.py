import logging
import sys

import numpy as np

import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)  # Filter out INFO & WARNING messages

from tensorflow.keras import datasets, layers, models, regularizers # type: ignore
import tensorflow.keras as keras


class Models(object):
    def __init__(self):
        tf.keras.backend.clear_session() # type: ignore
        self.first_build = True
        
    # TODO
    def custom_model(self, model):
        #User custom data load. Just test portion

        #\User custom data load
        if(self.first_build):
        #User custom model build

        #\User custom model build

            self.first_build = False

            #User load pretrained weights

            #\User load pretrained weights
        else:
            #User load pretrained weights

            #\User load pretrained weights

            pass #delete this statement when implemented

        return


    def cnn_mnist(self, model):
        num_classes = 10
        input_shape = (28, 28, 1)
        # Load the data and split it between train and test sets
        (_, _), (x_test, y_test) = datasets.mnist.load_data()
        # Scale images to the [0, 1] range
        x_test = x_test.astype("float32")/255.0
        # Make sure images have shape (28, 28, 1)
        x_test = np.expand_dims(x_test, -1)
        # convert class vectors to binary class matrices
        y_test = keras.utils.to_categorical(y_test, num_classes)

        if(self.first_build):
            print(f"\n|{'Building CNN-MNIST Model':=^100}|\n")

            model = models.Sequential()
            model.add(layers.Conv2D(32, kernel_size=(3, 3),
                    input_shape=input_shape, activation="relu"))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(layers.Flatten())
            model.add(layers.Dropout(0.5))
            model.add(layers.Dense(num_classes, activation="softmax"))
            model.compile(loss="categorical_crossentropy",
                        optimizer="adam", metrics=["accuracy"])
            
            self.first_build = False

            try:
                model.load_weights("./weights/weights_cnn_mnist.h5")
            except (FileNotFoundError, IOError):
                print("Problem loading the weights of the model")
                sys.exit()
        else:
            print(f"\n|{'Loading CNN-MNIST default weights':=^100}|\n")
            model.load_weights("./weights/weights_cnn_mnist.h5")

        return model, x_test, y_test


    def convnet_js(self, model):
        num_classes = 10
        input_shape = (32, 32, 3)
        (_, _), (x_test, y_test) = datasets.cifar10.load_data()
        x_test = x_test.astype("float32")/255.0
        y_test = y_test.astype("float32")/255.0

        if(self.first_build):
            print(f"\n|{'Building ConvNET-js Model':=^100}|\n")

            model = models.Sequential()
            model.add(layers.Conv2D(16, (5, 5),
                                    padding="same", activation='relu',
                                    input_shape=input_shape,
                                    kernel_regularizer=regularizers.L2(l2=0.0001),))
            model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(layers.Conv2D(20, (5, 5), strides=(1, 1), padding="same",
                    activation='relu', kernel_regularizer=regularizers.L2(l2=0.0001)))
            model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(layers.Conv2D(20, (5, 5), strides=(1, 1), padding="same",
                    activation='relu', kernel_regularizer=regularizers.L2(l2=0.0001)))
            model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(layers.Flatten())
            model.add(layers.Dense(num_classes, activation='softmax',
                    kernel_regularizer=regularizers.L2(l2=0.0001)))

            model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(), # type: ignore
                        metrics=['accuracy'])
            
            self.first_build = False

            try:
                model.load_weights("./weights/weights_convnet_js.h5")
            except (FileNotFoundError, IOError):
                print("Problem loading the weights of the model")
                sys.exit()
        else:
            print(f"\n|{'Loading ConvNET-js default weights':=^100}|\n")
            model.load_weights("./weights/weights_convnet_js.h5")

        return model, x_test, y_test
