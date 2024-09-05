# Fine-Tuning the MobileNet V3 Small Model for Violence Detection

This project provides a simple example of fine-tuning the MobileNet V3 small model for the task of violence detection. The MobileNet V3 architecture is chosen for its efficiency in real-time applications.

## Dataset
The dataset used for fine-tuning the model is sourced from [Kaggle](https://www.kaggle.com/datasets/abdulmananraja/real-life-violence-situations/data). This dataset can be organized into three subsets:
- **Training Set**: 70% of the total dataset
- **Validation Set**: 20% of the total dataset
- **Test Set**: 10% of the total dataset

This division allows for robust fine-tuning and evaluation of the model, ensuring that it generalizes well to unseen data. The following Python script can be used for this data preparation:

```python

import os
import shutil
import numpy as np

# The split ratios
split_ratios = [0.7, 0.2, 0.1]

# The dataset directory
base_dir = 'data_violence'
non_violence_dir = os.path.join(base_dir, 'non_violence')
violence_dir = os.path.join(base_dir, 'violence')

# The new directory
new_base_dir = 'dataset'
os.makedirs(new_base_dir, exist_ok=True)

# Creating subdirectories for train, validation, and test
for split in ['train', 'val', 'test']:
    for category in ['non_violence', 'violence']:
        os.makedirs(os.path.join(new_base_dir, split, category), exist_ok=True)

# Function to split dataset
def split_dataset(source_dir, split_ratios):
    # Getting all images in the directory
    images = os.listdir(source_dir)
    # Shuffling the images
    np.random.shuffle(images)
    # Calculating split indices
    total_images = len(images)
    train_end = int(split_ratios[0] * total_images)
    val_end = train_end + int(split_ratios[1] * total_images)
    # Splitting the dataset
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    return train_images, val_images, test_images

# Splitting non-violence dataset
non_violence_train, non_violence_val, non_violence_test = split_dataset(non_violence_dir, split_ratios)

# Splitting violence dataset
violence_train, violence_val, violence_test = split_dataset(violence_dir, split_ratios)

# Function to copy images to new directories
def copy_images(images, source_dir, dest_dir):
    for image in images:
        shutil.copy(os.path.join(source_dir, image), os.path.join(dest_dir, image))

# Copy images to the new directories
copy_images(non_violence_train, non_violence_dir, os.path.join(new_base_dir, 'train', 'non_violence'))
copy_images(non_violence_val, non_violence_dir, os.path.join(new_base_dir, 'val', 'non_violence'))
copy_images(non_violence_test, non_violence_dir, os.path.join(new_base_dir, 'test', 'non_violence'))

copy_images(violence_train, violence_dir, os.path.join(new_base_dir, 'train', 'violence'))
copy_images(violence_val, violence_dir, os.path.join(new_base_dir, 'val', 'violence'))
copy_images(violence_test, violence_dir, os.path.join(new_base_dir, 'test', 'violence'))

```

## Results
After fine-tuning the MobileNet V3 small model, the results are exported into the ```MobileNetV3-Violence-Detection/SavedModel``` directory. The following command can be used to generate the ```mobilenetv3_model.onnx``` file:

```
bash
python -m tf2onnx.convert --saved-model SavedModel --output mobilenetv3_model.onnx --opset 13
```

The performance of ```mobilenetv3_model.onnx``` on the test dataset is evaluated as follows:

Using the ```onnxruntime``` library for inference in ```MobileNetV3-Violence-Detection/onnxruntime_inference.py```, the results are as follows:

- **True Positives** (Violence): 496
- **True Negatives** (Non-Violence): 422
- **False Positives** (Predicted Violence, Actual Non-Violence): 102
- **False Negatives** (Predicted Non-Violence, Actual Violence): 89
- **Total Images:** 1109
- **Accuracy:** 82.78%

In the next test, the ```python tract``` library is used for inference on the same test dataset in ```MobileNetV3-Violence-Detection/tract_inference.py```, yielding the exact same results:

- **True Positives** (Violence): 496
- **True Negatives** (Non-Violence): 422
- **False Positives** (Predicted Violence, Actual Non-Violence): 102
- **False Negatives** (Predicted Non-Violence, Actual Violence): 89
- **Total Images:** 1109
- **Accuracy:** 82.78%
