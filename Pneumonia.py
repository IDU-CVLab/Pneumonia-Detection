#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 13:14:14 2023

@author: idu
"""
import numpy as np
import pandas as pd
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.utils import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence, to_categorical

import matplotlib.pyplot as plt
from PIL import Image
from termcolor import colored  
from tensorflow.keras.preprocessing.sequence import pad_sequences

#################################################################################################################
#################################### Check the COVID detection method on CT pnomonia Cases #######################
################################################### URL https:://github.com/IDU-CVLab/COV19D ###################
#################################################################################################################

## Load models weight [The Pretrained COVID CNN MODEL]
model.load_weights('/home/idu/Desktop/COV19D/models/imagepreprocesscnnclass.h5')

model = keras.models.load_model('/home/idu/Desktop/COV19D/saved-models/Pnumonia-imageprocess-augment-sliceremove-3L-16-32-64-D128-cnn.h5')


# Set the class probability threshold for prediction at slice level
threshold = 0.5 #default threshol

# Trying heigher threshold for Pnumonia Cases compared to COVID-19
threshold = 0.8 #default threshol

# Define empty matrices to store the predicted COVID and NON-COVID images
covid_images = []
non_covid_images = []

data_folder = '/home/idu/Desktop/COV19D/validation/non-covid/'


# Loop through the subfolders in the data folder [This is the directory for Comon Pnomonia is at]
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
        img=cv2.resize(img, (300, 227))
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
        print(pred)
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
        
 
        
 #################################################################################################################
 #################################### Replicating the method for three classes to include Common Pnomonia #######################
 #################################################################################################################
 
# test-59, 61, 62, and 63 were used for training from the CT dataset of pnomonia for training. The rest are used for validation
# Common Pnomonia folders were copied 28 times in the training folder (Oversampling-Duplication) to increase the class aiming at classes balance
# Slices were already processed as in our previous paper at https://www.tandfonline.com/doi/abs/10.1080/21681163.2023.2219765?journalCode=tciv20
# or the free preprint version at https://arxiv.org/abs/2111.11191


train_dir = '/home/idu/Desktop/COV19D/train/'
val_dir = '/home/idu/Desktop/COV19D/val2/'
train_dir_aug = '/home/idu/Desktop/COV19D/train/CP/'  # Augmented Using Augmentor method

img_height = 300 
img_width = 227
#img_height = img_width = 224
batch_size = 16

#### TRAINING GENERATOR WITH CLASS AUGMENTATION

# [1] Using Augmentator to augment Common Pnumonia with focus on flipping, zooming and rotation
# URL https://github.com/mdbloice/Augmentor

import Augmentor

p = Augmentor.Pipeline(train_dir_aug)
#Initialised with 199 image(s) found.
#Output directory set to /home/idu/Desktop/COV19D/train/CP/output.

# Choosing the augmentation method 
p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
#p.rotate270(probability=0.5)
p.flip_left_right(probability=0.3)
p.flip_top_bottom(probability=0.3)
#p.crop_random(probability=0.8, percentage_area=0.5)
p.zoom(probability=0.8, min_factor=1.1, max_factor=1.5)

# Adding samples to the minority calss ('CP')
p.sample(2000) 


# Define augmentation parameters with focus on flipping, zooming and rotation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
)


train_generator = train_datagen.flow_from_directory(
    train_dir, ## Images are already processed with cropping and slice removal
    target_size=(img_height, img_width),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    classes=['covid', 'non-covid', 'CP']
)

#### VALIDATION SET 

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    val_dir, ## Images are already processed with cropping and slice removal
    target_size=(img_height, img_width),
    color_mode='grayscale',
    #color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    classes=['covid', 'non-covid', 'CP']  # For Augmentation method
)


#### TRAINING THE CNN MODEL

from tensorflow.keras import models, layers, optimizers

## The standard Intial Model 
def make_model():
   
    model = models.Sequential()
    
    # Convulotional Layer 1
    model.add(layers.Conv2D(16,(3,3),input_shape=(img_height, img_width, 1), padding="same"))
    #model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))
    
    # Convulotional Layer 2
    model.add(layers.Conv2D(32,(3,3), padding="same"))  
    #model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))
    
    # Convulotional Layer 3
    model.add(layers.Conv2D(64,(3,3), padding="same"))
    #model.add(layers.BatchNormalization())
    model.add(layers.ReLU())   
    model.add(layers.MaxPooling2D((2,2)))
    
    # Convulotional Layer 4
    #model.add(layers.Conv2D(128,(3,3), padding="same"))
    #model.add(layers.BatchNormalization())
    #model.add(layers.ReLU())
    #model.add(layers.MaxPooling2D((2,2)))
    
    # Fully Connected Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(128))
    #model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.2))
    
    # Dense Layer  
    model.add(layers.Dense(3,activation='softmax'))
    
    
    return model

# Create the model
model = make_model()


from keras import metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', metrics.Precision(), metrics.Recall()]
)


early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=20)
checkpoint_cb = ModelCheckpoint('/home/idu/Desktop/COV19D/saved-models/Pnumonia-imageprocess-augment+-sliceremove-3L-16-32-64-D128-cnn-300-227.h5',
save_best_only=True, save_weights_only=False)


from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights
class_names = ['covid', 'non-covid', 'CP']
train_labels = train_generator.classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)

# Convert class weights to dictionary
class_weights_dict = dict(zip(np.unique(train_labels), class_weights))

# Convert class weights to dictionary
class_weights_dict = dict(zip(np.unique(train_labels), class_weights))

history = model.fit(train_generator, 
                    steps_per_epoch=train_generator.samples//batch_size, 
                    epochs=60, 
                    validation_data=val_generator, 
                    validation_steps=val_generator.samples//batch_size,
                    verbose=1, 
                    callbacks=[early_stopping_cb, checkpoint_cb],
                    class_weight=class_weights_dict,
                    )

#model.save('/home/idu/Desktop/COV19D/saved-models/Pnumonia-imageprocess-augment-sliceremove-4L-standard-cnn.h5')

########### Evaluation

model.evaluate(val_generator, batch_size=16)

print (history.history.keys())
            
Train_accuracy = history.history['accuracy']
print(Train_accuracy)
print(np.mean(Train_accuracy))
val_accuracy = history.history['val_accuracy']
print(val_accuracy)
print( np.mean(val_accuracy))

losss = history.history['loss']
print(losss)
print( np.mean(losss))
val_losss = history.history['val_loss']
print(val_losss)
print( np.mean(val_losss))


epochs = range(1, len(Train_accuracy)+1)
plt.figure(figsize=(12,6))
plt.plot(epochs, Train_accuracy, 'g', label='Training acc')
plt.plot(epochs, val_accuracy, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.ylim(0.45,1)
plt.xlim(0,50)
plt.legend()

plt.show()

val_recall = history.history['val_recall']
print(val_recall)
avg_recall = np.mean(val_recall)
avg_recall

val_precision = history.history['val_precision']
avg_precision = np.mean(val_precision)
avg_precision

epochs = range(1, len(Train_accuracy)+1)
plt.figure(figsize=(12,6))
plt.plot(epochs, val_recall, 'g', label='Validation Recall')
plt.plot(epochs, val_precision, 'b', label='Validation Prcision')
plt.title('Validation recall and Validation Percision')
plt.xlabel('Epochs')
plt.ylabel('Recall and Precision')
plt.legend()
plt.ylim(0,1)

plt.show()

Macro_F1score = (2*avg_precision*avg_recall) / (avg_precision + avg_recall)
Macro_F1score

precision = result.history['precision']
recall = result.history['recall']
f1_score = result.history['f1_score']

############################## Making Predictions on the validation set
## Choosing the directory where the test/validation data is at

folder_path = '/home/idu/Desktop/COV19D/val2/covid/'

covidd = []
noncovidd = []
cp = []

COVID = []
NONCOVID = []
CP = []
results =1
for fldr in os.listdir(folder_path):
    sub_folder_path = os.path.join(folder_path, fldr)
    for filee in os.listdir(sub_folder_path):
        file_path = os.path.join(sub_folder_path, filee)
        img =load_img(file_path, color_mode='grayscale')
        img=img_to_array(img)
        #img=cv2.resize(img, (300, 227))
        img=cv2.resize(img, (224, 224))
        img=img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.0
        
        ## Taking maximuim class probability
        resultt = model.predict(img)
        #print(resultt) 
        #result = np.argmax(resultt, axis=-1) 
        
        ## Increase the probability of class 1
        resultt[0][1] *= 0.3  
        result = np.argmax(resultt, axis=-1)


        #resutlt = result[0]
        #print(result)
        if result == 0:  # Class 0
           covidd.append(results)
        if result == 1:
           noncovidd.append(results)
        if result == 2:
           cp.append(results) 
        
    #print(sub_folder_path, end="\r \n")
    ## The majority voting at Patient's level
    if len(covidd) > len(noncovidd) and len (covidd) > len(cp):
      #print(fldr, colored("COVID", 'blue'), len(covidd), "to", len(noncovidd), "and", len(cp))
      COVID.append(fldr) 
    if len(noncovidd) > len(covidd) and len (noncovidd) > len(cp):
      #print(fldr, colored("NON-COVID", 'red'), len(noncovidd), "to", len(covidd), "and", len(cp))
      NONCOVID.append(fldr)
    if len(cp) >= len(covidd) and len(cp) >= len(noncovidd):
      #print (fldr, colored("Comon Pneumonia", 'green'), len(cp), "to", len(covidd), "and", len(noncovidd))
      CP.append(fldr)    
       
    covidd=[]
    noncovidd=[]
    cp=[]

#Checking the results
print(len(COVID))
print(len(NONCOVID))
print(len(CP))


##### BY KENAN MORANI