from keras import models
from keras import layers
from keras.initializers import  glorot_uniform


def identity_shortcut_block(x, filters, stage, block):
    # defining name basis
    conv_name_base = 'res' + str(stage) + str(block) + '_branch'
    bn_name_base = 'bn' + str(stage) + str(block) + '_branch'

    # saving the input value
    x_shortcut = x

    # first block of the main path
    x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1),
                      padding='same', name=conv_name_base + '2a',
                      kernel_initializer = glorot_uniform(seed=0))(x)
    x = layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    # second block of the main path
    x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1),
                      padding='same', name=conv_name_base + '2b',
                      kernel_initializer = glorot_uniform(seed=0))(x)
    x = layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    # adding shortcut value to main path and passing it through a relu activation
    x = layers.Add()([x, x_shortcut])
    x = layers.Activation('relu')(x)

    return x


def convolutional_shortcut_block(x, filters, stage, block):
    # defining name basis
    conv_name_base = 'res' + str(stage) + str(block) + '_branch'
    bn_name_base = 'bn' + str(stage) + str(block) + '_branch'

    # saving the input value
    x_shortcut = x

    # first block of the main path
    x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(2, 2),
                      padding='same', name=conv_name_base + '2a',
                      kernel_initializer = glorot_uniform(seed=0))(x)
    x = layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    # second block of the main path
    x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1),
                      padding='same', name=conv_name_base + '2b',
                      kernel_initializer = glorot_uniform(seed=0))(x)
    x = layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    # shortcut path
    x_shortcut = layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=(2, 2),
                               padding='valid', name=conv_name_base + '1',
                               kernel_initializer = glorot_uniform(seed=0))(x_shortcut)
    x_shortcut = layers.BatchNormalization(axis=3, name=bn_name_base + '1')(x_shortcut)

    # adding shortcut value to main path and passing it through a relu activation
    x = layers.Add()([x, x_shortcut])
    x = layers.Activation('relu')(x)

    return x


def build_resnet18(input_shape, num_classes):
    x_input = layers.Input(input_shape, name='input')

    # conv_1
    x = layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                      kernel_initializer = glorot_uniform(seed=0), name='conv_conv_1')(x_input)
    x = layers.BatchNormalization(axis=3, name='bn_conv_1')(x)
    x = layers.Activation(activation = 'relu')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # conv_2
    x = identity_shortcut_block(x, filters=64, stage=2, block=1)
    x = identity_shortcut_block(x, filters=64, stage=2, block=2)

    # conv_3
    x = convolutional_shortcut_block(x, filters=128, stage=3, block=1)
    x = identity_shortcut_block(x, filters=128, stage=3, block=2)

    # conv_4
    x = convolutional_shortcut_block(x, filters=256, stage=4, block=1)
    x = identity_shortcut_block(x, filters=256, stage=4, block=2)

    # conv_5
    x = convolutional_shortcut_block(x, filters=512, stage=5, block=1)
    x = identity_shortcut_block(x, filters=512, stage=5, block=2)

    # prediction
    x = layers.AveragePooling2D(pool_size=(2, 2), padding='same', name='gap')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(units=num_classes, activation='softmax', name='output')(x)

    return models.Model(inputs=x_input, outputs=x, name='resnet18')
