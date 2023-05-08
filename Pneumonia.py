#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 13:14:14 2023

@author: idu
"""

import os
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img



## Directory where Pneumonia CT image for trial is at 
file_path='/home/idu/Desktop/COV19D/Pneumonia/CT/COVID/test-5/IMG-0002-00078.jpg'


## Trying on one image slice
c=load_img(file_path, color_mode='grayscale')
c=img_to_array(c)
c=cv2.resize(c, (224, 224))
c=img_to_array(c)

c= np.expand_dims(c, axis=0)
c /= 255.0
result = model.predict(c)

print(result)
if  result < 0.4:
    print('covid')
else:
    print('normal')
    

# Trying on the full set 
# Define the data folder containing the subfolders with CT common Pneumonia images
data_folder='/home/idu/Desktop/COV19D/Pneumonia/CT/CP/'  ## Common Pneumonia CT images Directory


# Load the Keras model
from keras import models, layers

model = tf.keras.models.load_model('/home/idu/Downloads/OLD-CNN.h5')

model = tf.keras.models.load_model('/home/idu/Downloads/Image-process-sliceremove-cnn-class.h5')


## Adjusting input images height and weidth
h=w=224

def make_model():
   
    model = models.Sequential()
    
    # Convulotional Layer 1
    model.add(layers.Conv2D(16,(3,3),input_shape=(h,w,1), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))
    
    # Convulotional Layer 2
    model.add(layers.Conv2D(32,(3,3), padding="same"))  
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))
    
    # Convulotional Layer 3
    model.add(layers.Conv2D(64,(3,3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())   
    model.add(layers.MaxPooling2D((2,2)))
    
    # Convulotional Layer 4
    model.add(layers.Conv2D(128,(3,3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))
    
    # Fully Connected Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.1))
    
    # Dense Layer  
    model.add(layers.Dense(1,activation='sigmoid'))
    
    
    return model

model = make_model()

## Load models weight
model.load_weights('/home/idu/Desktop/COV19D/models/imagepreprocesscnnclass.h5')

# Set the class probability threshold for prediction at slice level
threshold = 0.5 #default threshol

# Trying heigher threshold for Pnumonia Cases compared to COVID-19
threshold = 0.8 #default threshol

# Define empty matrices to store the predicted COVID and NON-COVID images
covid_images = []
non_covid_images = []


# Loop through the subfolders in the data folder
for subdir in os.listdir(data_folder):
    sub_folder_path = os.path.join(data_folder, subdir)
    
    # Ignore any files in the data folder
    if not os.path.isdir(sub_folder_path):
        continue
    
    # Define counters for the number of predicted COVID and NON-COVID images
    covid_count = 0
    non_covid_count = 0
    
    # Loop through the image files in the subfolder
    for filee in os.listdir(sub_folder_path):
        file_path = os.path.join(sub_folder_path, filee)
        img =load_img(file_path, color_mode='grayscale')
        img=img_to_array(img)
        img=cv2.resize(img, (224, 224))
        #img=cv2.resize(img, (255, 298))
        #img=cv2.resize(img, (300, 512))
        img=img_to_array(img)

        img = np.expand_dims(img, axis=0)
        img /= 255.0
        
        # Load the image using cv2.imread
        #img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        #img = cv2.resize(img, (298, 255))
        #img = cv2.resize(img, (h, w))
        #img=load_img(file_path, color_mode='grayscale')
        #img /= 255.0
        
        # Resize the image to match the input shape of the model
        #img = cv2.resize(img, (256, 256))
        
        #img = np.expand_dims(img, axis=0)
        
        # Predict using the model
        
        #img=img_to_array(c)
        #img= np.expand_dims(img, axis=0)
        #img = np.expand_dims(img, axis=-1)
        pred = model.predict(img)[0][0]
        #print(pred)
        # Classify the image based on the prediction probability
        if pred < threshold:
            covid_images.append(file_path)
            covid_count += 1
        else:
            non_covid_images.append(file_path)
            non_covid_count += 1
    
    # Classify the subfolder based on the number of predicted COVID and NON-COVID images
    if covid_count >= non_covid_count:
        print(f"{subdir} is COVID")
    else:
        print(f"{subdir} is non-COVID")


### THE END
### KENAN MORANI
