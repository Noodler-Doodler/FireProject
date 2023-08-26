from tensorflow import keras

def get_model(input_shape: tuple):
    base_model = keras.applications.ResNet50(
        include_top=False,
        weights = "imagenet",
        input_shape=input_shape,
    )
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = keras.applications.resnet.preprocess_input(inputs)
    x = base_model(x, training=False)

    x = keras.layers.ReLU()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    output = keras.layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs,output)
