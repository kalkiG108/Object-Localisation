# Object-Localisation
This project involves building a Convolutional Neural Network (CNN) from scratch to perform two tasks using the MNIST dataset:

1. Classify the main subject (digit) in an image.
2. Localize the main subject by drawing bounding boxes around it.

PROJECT OVERVIEW

<img src="data:image/png;base64,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" width="300" />
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

