# Object-Localisation
This project involves building a Convolutional Neural Network (CNN) from scratch to perform two tasks using the MNIST dataset:

1. Classify the main subject (digit) in an image.
2. Localize the main subject by drawing bounding boxes around it.

PROJECT OVERVIEW

Task 1: Classification
The model will predict the class of the digit present in the image (0-9).

Task 2: Localization
The model will predict the bounding box coordinates around the digit. This is modeled as a regression task, where the model outputs numeric values representing the coordinates of the bounding box.

DATASET

MNIST Dataset
The MNIST dataset consists of 28x28 pixel grayscale images of handwritten digits (0-9). For this project, each digit image is placed on a 75x75 black canvas at random locations to create a custom dataset. The bounding box coordinates for each digit are calculated accordingly.

MODEL ARCHITECTURE

The model is implemented using TensorFlow and Keras. It consists of three main parts:

1. Feature Extractor: Convolutional and pooling layers to extract features from the image.
2. Classifier: Fully connected layers to classify the digit.
3. Bounding Box Regressor: Fully connected layers to predict the bounding box coordinates.

MODEL SUMMARY

Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 73, 73, 16)        160       
_________________________________________________________________
average_pooling2d (AveragePooling2D)  (None, 36, 36, 16)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 34, 34, 32)        4640      
_________________________________________________________________
average_pooling2d_1 (AveragePooling2D)  (None, 17, 17, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 15, 15, 64)        18496     
_________________________________________________________________
average_pooling2d_2 (AveragePooling2D)  (None, 7, 7, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 3136)              0         
_________________________________________________________________
dense (Dense)                (None, 128)               401536    
_________________________________________________________________
classification (Dense)       (None, 10)                1290      
_________________________________________________________________
bounding_box (Dense)         (None, 4)                 516       
=================================================================
Total params: 426,638
Trainable params: 426,638
Non-trainable params: 0

TRAINING

The model is trained using the Adam optimizer. The loss function for the classification task is categorical cross-entropy, and for the bounding box regression task, it is mean squared error (MSE).

TRAINING AND VALIDATION
The dataset is split into training and validation sets. The model is trained for 10 epochs, with the following results:

