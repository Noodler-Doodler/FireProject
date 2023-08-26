import Pretrained_xception
import Pretrained_resnet50
import Xception
import ResNet50
import tensorflow as tf
from tensorflow import keras
import os

print("num GPU's Available: ", len(tf.config.list_physical_devices("GPU")))

train, val = keras.utils.image_dataset_from_directory(
    directory="C:\FireProject\FireProject_1\ForestFireDataset",
    labels="inferred",
    label_mode="binary",
    color_mode="rgb",
    image_size=(250, 250),
    shuffle=True,
    seed=1337,
    validation_split=0.2,
    subset="both",
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

model = Xception.make_model(input_shape=[250, 250, 3],
                            num_classes=2,
                            include_dropout=True,
                            dropout_amount=0.4)
# Allow dynamic Tensorflow optimization
train = train.prefetch(tf.data.AUTOTUNE)
val = val.prefetch(tf.data.AUTOTUNE)

"""Optimizer parameters"""
learn_rate = 0.001
beta1 = 0.999
beta2 = 0.99
"""End optimizer parameters"""

optimizer = keras.optimizers.Nadam(
    learning_rate=learn_rate,
    beta_1=beta1,
    beta_2=beta2,
)


# log directory
log_folder = "logs/Nadam-dropout-0.4/"
# save directory
save_folder = "ModelSaves/Nadam-dropout-0.4/"
run = "lr"+str(learn_rate) + "beta1"+str(beta1) + "beta2"+str(beta2)

# create log and save directories
os.makedirs(log_folder, exist_ok=True)
os.makedirs(save_folder, exist_ok=True)
log_dir = os.path.join(log_folder,run)
save_dir = os.path.join(save_folder,run)
# instantiate keras callback methods
csv_logger = keras.callbacks.CSVLogger(log_dir, separator=",", append=False)
save_best = keras.callbacks.ModelCheckpoint(
    save_dir,
    monitor="val_loss",
    save_best_only=True,
)

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train,
    epochs=5,
    callbacks=[csv_logger, save_best],
    validation_data=val,
)