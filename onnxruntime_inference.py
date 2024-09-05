import os
import cv2
import numpy as np
import onnxruntime


input_size = (224, 224)
model_path = 'mobilenetv3_model.onnx'
test_directory = 'dataset/test/'

# Loading the model
session = onnxruntime.InferenceSession(model_path)

def preprocess_image(image_path):
    # Loading
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    
    # Resizing
    image = cv2.resize(image, input_size)    
    # Normalizing
    image = image.astype(np.float32) / 255.0    
    # Add batch dimension [1, 224, 224, 3]
    image = np.expand_dims(image, axis=0)    
    return image

def predict(image):
    # Getting the model input name
    input_name = session.get_inputs()[0].name
    # Changing 'image' to [1, 224, 224, 3] format
    result = session.run(None, {input_name: image})
    return result

def test_model(test_directory):
    # Initialize counters
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    total_images = 0
    
    for class_dir in os.listdir(test_directory):
        class_path = os.path.join(test_directory, class_dir)
        if os.path.isdir(class_path):
            # Assign numeric labels based on directory names
            actual_class = 1 if class_dir == 'violence' else 0
            
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                
                # Preprocessing the image
                image = preprocess_image(image_path)
                # Making predictions
                result = predict(image)                
                # Getting the predicted class
                predicted_class = np.argmax(result[0])  # 0 or 1
                
                # Updating counters based on prediction
                if predicted_class == 1:   # Predicted violence
                    if actual_class == 1:  # Actual violence
                        true_positives += 1
                    else:  # Actual non-violence
                        false_positives += 1
                else:  # Predicted non-violence
                    if actual_class == 0:  # Actual non-violence
                        true_negatives += 1
                    else:  # Actual violence
                        false_negatives += 1
                
                total_images += 1

    # Calculating accuracy
    accuracy = (true_positives + true_negatives) / total_images if total_images > 0 else 0
    # Print results
    print(f"True Positives (Violence): {true_positives}")
    print(f"True Negatives (Non-Violence): {true_negatives}")
    print(f"False Positives (Predicted Violence, Actual Non-Violence): {false_positives}")
    print(f"False Negatives (Predicted Non-Violence, Actual Violence): {false_negatives}")
    print(f"Total Images: {total_images}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

# Run the test
test_model(test_directory)
