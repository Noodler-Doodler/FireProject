import os
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import csv

"""Adjustable parameters"""
model = "ModelSaves/XceptionNoDropout/Nadam"  # path to model save file
result_directory = "test_results/Xception/NoDropNadam" # test result save directory
# paths to positive and negative labeled images
pos_path = r"ForestFireDataset/Testing/fire"
neg_path = r"ForestFireDataset/Testing/nofire"
prediction_threshold = .5
image_size = (250, 250)
"""End adjustable parameters"""

os.makedirs(result_directory, exist_ok=True) # Check result_directory exists

test_data = keras.utils.image_dataset_from_directory(
    directory="ForestFireDataset/Testing",
    labels="inferred",
    label_mode="binary",
    color_mode="rgb",
    image_size=image_size,
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

actual = []  # list for actual labels
predicted = []  # list for predicted labels
FalsePos = []  # list for missed image file names to save
FalseNeg = []  # list for missed image file names to save
ConfidenceScores = []  # list for confidence scores for each prediction

# Keras built in evaluation
reconstructed_model = keras.models.load_model(model)
eval_loss, eval_acc = reconstructed_model.evaluate(test_data)
print(f"Evaluation accuracy {eval_acc}, Evaluation loss {eval_loss}")

# Custom prediction through each image
for pos_image in os.listdir(pos_path):
    image_path = pos_path + "/" + pos_image
    pos_image = keras.preprocessing.image.load_img(image_path, target_size=image_size)
    pos_image = keras.preprocessing.image.img_to_array(pos_image)
    pos_image = np.expand_dims(pos_image, axis=0)
    prediction = reconstructed_model.predict(pos_image)
    actual.append("1")
    ConfidenceScores.append(((1-prediction * 100),image_path))
    if prediction[0] < prediction_threshold:
        predicted.append("1")
    else:
        FalseNeg.append(image_path)
        predicted.append("0")

for neg_image in os.listdir(neg_path):
    image_path = neg_path + "/" + neg_image
    neg_image = keras.preprocessing.image.load_img(image_path, target_size=image_size)
    neg_image = keras.preprocessing.image.img_to_array(neg_image)
    neg_image = np.expand_dims(neg_image, axis=0)
    prediction = reconstructed_model.predict(neg_image)
    actual.append("0")
    ConfidenceScores.append(((prediction*100), image_path))
    if prediction > prediction_threshold:
        predicted.append("0")
    else:
        FalsePos.append(image_path)
        predicted.append("1")

# Create CSV with missed image paths
with open(os.path.join(result_directory, "missed.csv"), "w", newline="") as file:
    writer = csv.writer(file)
    for image in FalsePos:
        writer.writerow([image])
    for image in FalseNeg:
        writer.writerow([image])

# create CSV with confidence (prediction) scores
with open(os.path.join(result_directory, "ConfidenceScores.csv"), "w", newline="") as file:
    writer = csv.writer(file)
    for score in ConfidenceScores:
        writer.writerow(score)

# Create confusion matrix based on actual predictions vs actual labels
confusion_matrix = metrics.confusion_matrix(actual, predicted,)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=["No-Fire", "Fire"],)
cm_display.plot(cmap="Blues")
plt.savefig(os.path.join(result_directory, "CM.jpg"))

# Create CSV with performance metrics
with open(os.path.join(result_directory, "metrics.csv"), "w", newline="") as file:
    file.write(f"Accuracy: {(metrics.accuracy_score(actual,predicted))}\n")
    file.write(f"f1-score: {metrics.f1_score(actual,predicted,pos_label='1')}\n")
    file.write(f"Precision: {metrics.precision_score(actual, predicted, pos_label='1')}\n")
    file.write(f"Recall: {metrics.recall_score(actual,predicted, pos_label='1')}\n")

print(f"Accuracy: {(metrics.accuracy_score(actual,predicted))}")
print(f"f1-score: {metrics.f1_score(actual,predicted,pos_label='1')}")
print(f"Precision: {metrics.precision_score(actual, predicted, pos_label='1')}")
print(f"Recall: {metrics.recall_score(actual,predicted, pos_label='1')}")