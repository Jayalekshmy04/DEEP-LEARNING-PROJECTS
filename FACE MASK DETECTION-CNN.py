IMPORTING THE DEPENDENCIES

import os  # for accessing files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2  # for image prosessing
from google.colab.patches import cv2_imshow
from PIL import Image #PIL -> pillow libary used for image processing libraries
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')

 #extracting the compessed Dataset
from zipfile import ZipFile
dataset='/content/drive/MyDrive/FaceMask (1) (1).zip'

with ZipFile(dataset,'r') as zip:  # r for reading the file
  zip.extractall()
  print('the dataset is extacted')

with_mask_files=os.listdir('/content/data/with_mask') # create list contain files with mask
print(with_mask_files[:5])  # print the first five elements
print(with_mask_files[-5:]) # print the last five elements of the list

without_mask_files=os.listdir('/content/data/without_mask') # create list contain files without mask
print(without_mask_files[:5])  # print the first five elements
print(without_mask_files[-5:])  # print the last five elements of the list

print('Number of mask images:',len(with_mask_files))
print('number of without mask images:', len(without_mask_files))

Creating lables for the two class of images

with masks-> 0


withoutmask -> 1


#creating the labels
with_mask_labels = [0]*3725
without_mask_labels = [1]*3828

print(with_mask_labels[:5])
print(without_mask_labels[:5])

print(len(with_mask_labels))
print(len(without_mask_labels))

labels = with_mask_labels + without_mask_labels # adding the to list

print(len(labels))
print(labels[0:5])
print(labels[-5:])

Displaying the images


# dispolaying with mask image
img=mpimg.imread('/content/data/with_mask/with_mask_1000.jpg') # read the image into numpy array
imgplot = plt.imshow(img)
plt.show()

#displaying without mask image
img = mpimg.imread('/content/data/without_mask/without_mask_1004.jpg')
imgplot = plt.imshow(img)
plt.show()

Image Processing

Resize the images

image processing

1.Resize the images


2.Convert the images to numpy arrays

#convert images to numpy arrays
with_mask_path = '/content/data/with_mask/' # don't forgot to add /
data = [] # create a empty list


for img_file in os.listdir(with_mask_path): #Iterate over files in the "with_mask_path" directory
    img_path = os.path.join(with_mask_path, img_file)
    if os.path.isfile(img_path): # check if "img_path" is a file
       image = Image.open(with_mask_path + img_file)  # open the all file example  /content/data/with_mask/with_mask_894.jpg
       image = image.resize((128,128)) #convert all the images  dimensions to 128 x 128
       image = image.convert('RGB')    # converting all the images to RGB
       image = np.array(image)         # converting to numpy array
       data.append(image)              # adding image to data


without_mask_path = '/content/data/without_mask/'
for img_file in os.listdir(without_mask_path): #Iterate over files in the "with_mask_path" directory
    img_path = os.path.join(without_mask_path ,img_file)
    if os.path.isfile(img_path): # check if "img_path" is a file
       image = Image.open(without_mask_path + img_file)  # open the all file example  /content/data/with_mask/with_mask_894.jpg
       image = image.resize((128,128)) #convert all the images  dimensions to 128 x 128
       image = image.convert('RGB')    # converting all the images to RGB
       image = np.array(image)         # converting to numpy array
       data.append(image)              # adding image to data


data[0]

type(data[0])

data[0].shape

#converting image list and label list to numpy arrays

X = np.array(data)
Y = np.array(labels)

type(X)

type(Y)

Y

Train test Split

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=2)

print(X.shape, x_train.shape, x_test.shape)

#scaling the data

x_train_scaled =  x_train/255   #to change the value from 0 to 1


x_test_scaled = x_test/255

x_train_scaled[0]

Building a convolutional Netural Networks (CNN)

import tensorflow as tf
from tensorflow import keras

num_of_classes = 2
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(128,128,3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(128,128,3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128, activation ='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(num_of_classes, activation='sigmoid'))

#compile the nerual network
model.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['acc'])

#training the neural network
history = model.fit(x_train_scaled,y_train, validation_split=0.1, epochs=5)

model evaluation

loss,accuracy = model.evaluate(x_test_scaled,y_test)
print('Test Accuracy= ', accuracy)

h = history

#plot the loss value
plt.plot(h.history['loss'], label='train loss')
plt.plot(h.history['val_loss'], label='validation loss')
plt.legend()
plt.show()

#plot the accuracy value
plt.plot(h.history['acc'], label='train accuracy')
plt.plot(h.history['val_acc'], label='vaildation accuracy')
plt.legend()
plt.show()

predictive System

# @title Default title text
input_image_path = input('path of the image to be pedicted: ')
input_image = cv2.imread(input_image_path)

cv2_imshow(input_image)

input_image_resized = cv2.resize(input_image, (128,128))

input_image_scaled = input_image_resized/255

input_image_reshaped = np.reshape(input_image_scaled,[1,128,128,3])

input_prediction=model.predict(input_image_reshaped)

print(input_prediction)

input_pred_label = np.argmax(input_prediction)

print(input_pred_label)

if input_pred_label == 1:
  print('The person in the image is not wearing a mask')

else:
  print('The person in the image is  wearing a mask')