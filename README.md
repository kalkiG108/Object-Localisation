# Object-Localisation
This project involves building a Convolutional Neural Network (CNN) from scratch to perform two tasks using the MNIST dataset:

1. Classify the main subject (digit) in an image.
2. Localize the main subject by drawing bounding boxes around it.

PROJECT OVERVIEW
data:image/png;base64,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
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

