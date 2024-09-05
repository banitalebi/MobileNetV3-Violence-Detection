import os
import cv2
import tract
import numpy as np


input_size = (224, 224)
model_path = 'mobilenetv3_model.onnx'
model = (
        tract.onnx()
        .model_for_path(model_path)
        .into_optimized()
        .into_runnable()
    )

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, input_size)
    image = image.astype(np.float32) / 255.0 
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, (0, 1, 2, 3))
    image = np.array(image)
    return image

def predict(image):
    result = model.run([image])
    return result[0].to_numpy()[0]


true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0
total_images = 0

test_directory = 'dataset/test/violence'
for image_path in os.listdir(test_directory):
    image_path = os.path.join(test_directory, image_path)
    confidences = predict(preprocess_image(image_path))
    predicted_class = np.argmax(confidences)
    if predicted_class == 1:
        true_positives+=1
    else:
        false_negatives+=1
    total_images+=1

test_directory = 'dataset/test/non_violence'
for image_path in os.listdir(test_directory):
    image_path = os.path.join(test_directory, image_path)
    confidences = predict(preprocess_image(image_path))
    predicted_class = np.argmax(confidences)
    if predicted_class == 0:
        true_negatives+=1
    else:
        false_positives+=1
    total_images+=1

accuracy = (true_positives + true_negatives) / total_images if total_images > 0 else 0

print(f"True Positives (Violence): {true_positives}")
print(f"True Negatives (Non-Violence): {true_negatives}")
print(f"False Positives (Predicted Violence, Actual Non-Violence): {false_positives}")
print(f"False Negatives (Predicted Non-Violence, Actual Violence): {false_negatives}")
print(f"Total Images: {total_images}")
print(f"Accuracy: {accuracy * 100:.2f}%")