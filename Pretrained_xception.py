from tensorflow import keras

def get_model(input_shape):
    base_model = keras.applications.Xception(
        weights="imagenet",
        input_shape=input_shape,
        include_top=False,
    )
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1./255)(inputs)
    x = base_model(x,training=False)

    x = keras.layers.ReLU()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    output = keras.layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs,output)
