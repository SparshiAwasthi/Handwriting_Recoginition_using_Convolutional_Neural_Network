# Handwriting_Recognition_System_using_Convolutional_Neural_Network
The Handwriting Recognition System built using Convolutional Neural Networks(CNNs) with Tensorflow and Keras libraries demonstrates a powerful approach to recognize handwritten digits.

## The dataset

The MNIST database (Modified National Institute of Standards and Technology database)

- Number of Samples: 70,000
- Contents: 60,000 training images and 10,000 test images
- Image Size: Each image is 28x28 pixels
- Labels: Digits from 0 to 9

## The Workflow of model

- Data Preparation
- Displaying a sample image
- Model architecture
- Model compilation and training
- Model evaluation

## Features

- Convolutional Neural Network (CNN): The core machine learning architecture used in the model for feature extraction and classification.
  
- Softmax Activation Function: Applied in the output layer to convert the model's raw scores into class probabilities, it's a fundamental component of many classification models.
  
- Rectified Linear Unit (ReLU): ReLU is an activation function used in the neural network layers to introduce non-linearity.

- Categorical Cross-Entropy Loss: The loss function used for training the model, which measures the error between predicted and actual labels.

- Adam Optimizer: A gradient-based optimization algorithm used to update the model's weights during training.

- Dropout Regularization: A regularization technique used to prevent overfitting during training by randomly dropping a fraction of neurons.

## Dependencies

### Libraries:
- TensorFlow: The core machine learning framework used for building and training neural networks.
- Matplotlib: Used for data visualization, particularly for displaying sample images.
- NumPy (not explicitly imported but likely used internally): A fundamental library for numerical computations. It is often used internally in TensorFlow operations.

### Datasets:
- MNIST Dataset: The model preprocesses the MNIST dataset, which includes handwritten digit images and their corresponding labels. The dataset is split into training and test sets.
