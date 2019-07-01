from tensorflow.python.keras import models
from tensorflow.python.keras import layers


def deep(features_shape, number_of_classes, activation_function='relu'):
    model = models.Sequential()

    # Input
    model.add(layers.InputLayer(input_shape=features_shape, name='Inputs', dtype='float32'))

    # Flatten
    model.add(layers.Flatten(name='Flatten'))

    # Dense block
    model.add(layers.Dense(units=512, activation=activation_function, name='Dense1'))
    model.add(layers.Dense(units=512, activation=activation_function, name='Dense2'))
    model.add(layers.Dense(units=512, activation=activation_function, name='Dense3'))

    # Predictions
    model.add(layers.Dense(units=number_of_classes, activation=activation_function, name='Prediction'))

    # Print network summary
    model.summary()

    return model


def deep_cnn(features_shape, num_classes, activation_function='relu'):
    model = models.Sequential()

    model.add(layers.InputLayer(input_shape=features_shape, name='Inputs', dtype='float32'))

    # Block 1
    model.add(layers.Conv2D(input_shape=features_shape,
                            filters=32, kernel_size=(3, 3), strides=1,
                            padding='same', activation=activation_function,
                            name='Block1_Convolution'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                  padding='same',
                                  name='Block1_MaxPooling'))
    model.add(layers.BatchNormalization(name='Block1_BatchNormalization'))

    # Block 2
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1,
                            padding='same', activation=activation_function,
                            name='Block2_Convolution'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                  padding='same',
                                  name='Block2_MaxPooling'))
    model.add(layers.BatchNormalization(name='Block2_BatchNormalization'))

    # Block 3
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1,
                            padding='same', activation=activation_function,
                            name='Block3_Convolution'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                  padding='same',
                                  name='Block3_MaxPooling'))
    model.add(layers.BatchNormalization(name='Block3_BatchNormalization'))

    # Flatten
    model.add(layers.Flatten(name='Flatten'))

    # Dense block
    model.add(layers.Dense(units=64, activation=activation_function, name='Dense_Dense'))
    model.add(layers.BatchNormalization(name='Dense_BatchNormalization'))
    model.add(layers.Dropout(rate=0.2, name='Dense_Dropout'))

    # Predictions
    model.add(layers.Dense(units=num_classes, activation='softmax', name='Predictions_Dense'))

    # Print network summary
    model.summary()

    return model
