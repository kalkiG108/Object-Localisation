# Object-Localisation
This project involves building a Convolutional Neural Network (CNN) from scratch to perform two tasks using the MNIST dataset:

Classify the main subject (digit) in an image.
Localize the main subject by drawing bounding boxes around it.
Project Overview
Task 1: Classification
The model will predict the class of the digit present in the image (0-9).

Task 2: Localization
The model will predict the bounding box coordinates around the digit. This is modeled as a regression task, where the model outputs numeric values representing the coordinates of the bounding box.

Dataset
MNIST Dataset
The MNIST dataset consists of 28x28 pixel grayscale images of handwritten digits (0-9). For this project, each digit image is placed on a 75x75 black canvas at random locations to create a custom dataset. The bounding box coordinates for each digit are calculated accordingly.

Model Architecture
The model is implemented using TensorFlow and Keras. It consists of three main parts:

1. Feature Extractor: Convolutional and pooling layers to extract features from the image.
2. Classifier: Fully connected layers to classify the digit.
3. Bounding Box Regressor: Fully connected layers to predict the bounding box coordinates.
