import keras
import keras_cv
from keras.layers import Input, Conv2D, SeparableConv2D, \
    Add, Dense, ReLU, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization , Dropout


def conv_with_bn(x, filters, kernel_size, strides=1):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    return x


def sep_with_bn(x, filters, kernel_size, strides=1):
    x = SeparableConv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    return x




def make_model(input_shape, num_classes, include_dropout, dropout_amount=0):
    """
    :param input_shape: Image input size parameters
    :param num_classes: Number of classes
    :param include_dropout: Whether or not to include dropout
    :param dropout_amount: Amount of dropout to be included if include_dropout is True. 0-1 as percentage
    :return: Keras Model object
    """
    inputs = Input(shape=input_shape)
    x = keras.layers.Rescaling(1. / 255)(inputs)

    #entry flow
    x = conv_with_bn(x, filters=32, kernel_size=3, strides=2)
    x = ReLU()(x)
    x = conv_with_bn(x, filters=64, kernel_size=3)
    residual = ReLU()(x)

    x = sep_with_bn(residual, filters=128, kernel_size=3)
    x = ReLU()(x)
    x = sep_with_bn(x, filters=128, kernel_size=3)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    residual = conv_with_bn(residual, filters=128, kernel_size=1, strides=2)

    x = Add()([residual, x])
    x = ReLU()(x)
    x = sep_with_bn(x, filters=256, kernel_size=3)
    x = ReLU()(x)
    x = sep_with_bn(x, filters=256, kernel_size=3)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    residual = conv_with_bn(residual, filters=256, kernel_size=1, strides=2)

    x = Add()([residual, x])
    x = ReLU()(x)
    x = sep_with_bn(x, filters=728, kernel_size=3)
    x = ReLU()(x)
    x = sep_with_bn(x, filters=728, kernel_size=3)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    residual = conv_with_bn(residual, filters=728, kernel_size=1, strides=2)
    tensor = Add()([residual, x])

    #middle flow
    for _ in range(8):  # middle flow repeats 8 times
        x = ReLU()(tensor)
        x = sep_with_bn(x, filters=728, kernel_size=3)
        x = ReLU()(x)
        x = sep_with_bn(x, filters=728, kernel_size=3)
        x = ReLU()(x)
        x = sep_with_bn(x, filters=728, kernel_size=3)

        tensor = Add()([tensor, x])

    #exit flow
    x = ReLU()(tensor)
    x = sep_with_bn(x, filters=728, kernel_size=3)
    x = ReLU()(x)
    x = sep_with_bn(x, filters=1024, kernel_size=3)
    x = MaxPooling2D(3, strides=2, padding='same')(x)

    tensor = conv_with_bn(tensor, filters=1024, kernel_size=1, strides=2)

    x = Add()([tensor, x])
    x = sep_with_bn(x, filters=1536, kernel_size=3)
    x = ReLU()(x)
    x = sep_with_bn(x, filters=2048, kernel_size=3)
    x = ReLU()(x)
    x = GlobalMaxPooling2D()(x)

    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes
    # ---
    if include_dropout:
        x = Dropout(dropout_amount)(x)
    outputs = Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

