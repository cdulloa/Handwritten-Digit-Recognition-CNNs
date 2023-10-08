# Handwritten Digit Recognition

This is a project that uses convolutional neural networks (CNNs) to recognize handwritten digits from the MNIST dataset. It can also load and predict new images of digits using the trained model.

## Installation

To run this project, you need to have Python 3 and the following libraries installed:

- Keras
- TensorFlow
- h5py
- numpy

You can install them using pip or conda commands. For example:

```bash
pip install keras tensorflow h5py numpy
```
## Usage

To use this project, follow these steps:

1. Clone or download this repository to your local machine.
2. Open a terminal or command prompt and navigate to the folder containing the `digitRec.py` file.
3. Run the `digitRec.py` file using the following command:

   ```bash
   python digitRec.py
   ```
The script performs the following tasks:

- Loads and preprocesses the MNIST dataset.
- Defines and compiles the CNN model.
- Trains the model on the training data.
- Evaluates the model's performance on the test data.
- Saves the trained model to a file named `digit_recognizer.h5`.

Additionally, the script can:

- Load a new image of a digit from the `sample_images` folder.
- Preprocess the image.
- Predict the digit's class using the trained model.
- Print the predicted class in the terminal or command prompt.

You can modify the script to load and predict different images of digits or experiment with different model parameters and hyperparameters.

## Features

This project includes the following features:

- Utilizes Convolutional Neural Networks (CNNs) to recognize handwritten digits from the MNIST dataset, making it suitable for image classification tasks.
- Utilizes Keras, a high-level neural network API that runs on top of TensorFlow, to define and train the CNN model. Keras simplifies neural network development in Python.
- Utilizes h5py, a Python package that provides an interface to the HDF5 binary data format, to save and load the trained model. HDF5 is a popular format for storing complex data structures efficiently.
- Utilizes numpy, a Python package for scientific computing, to manipulate arrays and matrices. Numpy is commonly used for numerical and mathematical operations in Python.
