from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization , Flatten,\
    Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,MaxPool2D, SeparableConv2D, ReLU
from keras.models import Model

import tensorflow as tf
from tensorflow import keras

def identity_block(x, f, filters):
    F1, F2, F3 = filters #unpack filters

    x_shortcut = x

    x = Conv2D(filters=F1, kernel_size=(1,1), strides=(1,1),padding="valid")(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)

    x = Conv2D(filters=F2, kernel_size=(f,f), strides=(1,1),padding="same")(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)

    x = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1),padding="valid")(x)
    x = BatchNormalization(axis=3)(x)

    x = Add()([x, x_shortcut]) #skip connection
    x = Activation("relu")(x)

    return x

def convolutional_block(x, f, filters, s=2):
    F1, F2, F3 = filters

    x_shortcut = x

    x = Conv2D(filters=F1, kernel_size=(1,1), strides=(s,s), padding="valid")(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)

    x = Conv2D(filters=F2, kernel_size=(f,f), strides=(1, 1), padding="same")(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)

    x = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid")(x)
    x = BatchNormalization(axis=3)(x)

    x_shortcut = Conv2D(filters=F3, kernel_size=(1,1), strides=(s,s), padding="valid")(x_shortcut)
    x_shortcut = BatchNormalization(axis=3)(x_shortcut)

    x = Add()([x, x_shortcut])
    x = Activation("relu")(x)

    return x


def make_model(include_dropout: bool=None, dropout_amount: float = 0.0, input_shape: tuple =(250,250,3)):
    """
    :BOOL include_dropout: Whether or not to include dropout
    :Float dropout_amount: Amount of dropout to be included if include_dropout is True. 0-1 as percentage
    :Tuple input_shape: Shape of input to the model - default 250,250,3 for RGB values
    :return: Keras Model object
    """
    x_input = Input(input_shape)
    x_input = keras.layers.Rescaling(1./255)(x_input)
    x = ZeroPadding2D((3,3))(x_input)

    x = Conv2D(64, (7,7), strides=(2,2))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((3,3), strides=(2,2))(x)

    x = convolutional_block(x, f=3, filters=[64, 64, 256], s=1)
    x = identity_block(x, f=3, filters=[64, 64, 256])
    x = identity_block(x, f=3, filters=[64, 64, 256])

    x = convolutional_block(x, f=3, filters=[128,128,512], s=2)
    x = identity_block(x, f=3, filters=[128, 128, 512])
    x = identity_block(x, f=3, filters=[128, 128, 512])
    x = identity_block(x, f=3, filters=[128, 128, 512])

    x = convolutional_block(x, f=3, filters=[256, 256, 1024], s=2)
    x = identity_block(x, f=3, filters=[256, 256, 1024])
    x = identity_block(x, f=3, filters=[256, 256, 1024])
    x = identity_block(x, f=3, filters=[256, 256, 1024])
    x = identity_block(x, f=3, filters=[256, 256, 1024])
    x = identity_block(x, f=3, filters=[256, 256, 1024])

    x = convolutional_block(x, f=3, filters=[512, 512, 2048], s=2)
    x = identity_block(x, f=3, filters=[512,512,2048])
    x = identity_block(x, f=3, filters=[512, 512, 2048])

    x = AveragePooling2D(pool_size=(2,2), padding="same")(x)
    x = Flatten()(x)
    if include_dropout:
        x = keras.layers.Dropout(dropout_amount)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=x_input, outputs=outputs, name="ResNet50")
    return model