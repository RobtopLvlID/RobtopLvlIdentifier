![Preview](static/logo.png)  


---


<div align="center">
  <h2>Model capable of identifying all levels of Robtop (Geometry Dash :p) 🤖</h2> 
  <img src="static/robtop.png" alt="Preview" />
</div>

---

## EN🇬🇧 
### Development process📝  
A few months ago, as I began exploring the field of artificial intelligence, I wondered whether it would be possible to develop a model capable of recognizing Geometry Dash levels. My goal was not to exploit this for unfair advantage in Sparky, especially considering the evident hardware limitations—even when utilizing cloud resources. Ultimately, I decided to base the project on the pre-trained MobileNetV2 model from Keras, leveraging convolutional neural networks for image recognition. The complete code can be found in the repository [GDLvlDetector](https://github.com/ANGELUSD11/GDLvlDetector/). Initially, I attempted to build the network from scratch using TensorFlow, which surprisingly performed quite well in recognizing around two to five levels at most. However, the primary challenge with this approach was the sheer volume of data required and the hardware constraints. In the following sections, I will explain each phase of the training process I undertook.  

---

- Step 1: Data Collection  
As with any machine learning project, the initial step that must be taken—and the one that will ultimately determine the quality of the model—is gathering the training data. For this project, I spent several days recording all the levels that the model would later recognize using images. However, I could not train the model using video footage directly, which leads us to the second step. 

- Step 2: Frame Extraction  
The proper way to train a convolutional neural network is by using images. Therefore, I extracted frames from the videos I had recorded earlier. Naturally, I didn’t do this manually but instead used OpenCV with the following script:  
```python
import os
import cv2

def extract_frames(video_path, output_folder, interval=30):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    saved_frames = 0

    while cap.isOpened(): 
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            filename = os.path.join(output_folder, f"{os.path.basename(video_path).split('.')[0]}_frame_{saved_frames}.jpg")
            cv2.imwrite(filename, frame)
            saved_frames += 1

        frame_count += 1

    cap.release()
```
- Step 3: Data Preparation
- 
In the previous step, the images from the frames were saved in an orderly manner into folders named after each level. Now, it is necessary to prepare the data in the most optimal way for use with the MobileNetV2 model, so that during the convolution process it can classify and recognize patterns correctly. This involves taking several aspects into account, such as image size, batch size, vector sizes, color channels, and many other tedious but essential details. The following script details this process:
```python
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
from keras.src.applications.mobilenet_v2 import preprocess_input, MobileNetV2
from keras.src import layers, models

# Path to the train data
data_path = "vidframes"

# Callbacks
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint("best_model.keras", save_best_only=True)
]

# Data Augmentation + Preprocess MobileNetV2
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

train_data = datagen.flow_from_directory(
    data_path,
    target_size=(160, 160),  # Size for MobileNetV2
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_path,
    target_size=(160, 160),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
```
