# Artificial Neural Network Ensemble for Predicting the Chezy Roughness Coefficient

This repository contains the software implementation of a computational algorithm based on an ensemble of neural networks designed for predicting the Chezy roughness coefficient. The ensemble consists of three identical neural network models and is optimized for approximate predictions in hydrodynamics and fluid mechanics.

## Features

•	Homogeneous ensemble of three identical neural network models.
•	The base neural network model is a fully connected feedforward neural network with a single hidden layer; the output neuron is linear, while the hidden layer neurons use a sigmoid activation function.
•	Utilizes independent datasets for training each neural network.
•	Training of the ensemble follows the bagging method (Bootstrap Aggregating).
•	Uses the backpropagation method for neural network training.
•	Aggregates the predictions of the three neural networks into a single forecasting result using the majority voting method, considering inverse problem solutions.
•	Predicts the Chezy roughness coefficient, a crucial parameter in hydraulics.
•	Intended for use in computational hydrodynamics and related fields.
•	Implemented using modern machine learning techniques in Python (version 3.10.11).

## Main Components

•	C_EANN_training.py - A Python module implementing the computational algorithm for training the neural network ensemble ANN_A, ANN_B1, and ANN_B2. Uses data from Training_Data.xlsx.
•	C_EANN_calculating.py - A Python module implementing the computational algorithm for predicting the Chezy roughness coefficient using the ensemble of neural networks. Uses data from Input.xlsx, weights_matrix_1_A.txt, weights_matrix_2_A.txt, weights_matrix_1_B1.txt, weights_matrix_2_B1.txt, weights_matrix_1_B2.txt, and weights_matrix_2_B2.txt.
•	Training_Data.xlsx - An Excel data file containing training and test examples for the neural networks ANN_A, ANN_B1, and ANN_B2. It also specifies the main parameters of the neural networks: number of inputs, number of hidden layer neurons, and number of outputs. A sample file is located in the Data folder of this repository. For proper operation of C_EANN_training.py, training data files must follow this format.
•	weights_matrix_1_[network_id].txt, weights_matrix_2_[network_id].txt - Text files containing weight matrices of trained neural networks with identifiers A, B1, or B2. These files are created in the Data folder as a result of executing C_EANN_training.py.
•	Input.xlsx - An Excel data file containing input data vectors (normalized and non-normalized hydromorphometric parameter values) for which the Chezy roughness coefficient needs to be computed. It also contains information about the main parameters of the neural networks, aligned with Training_Data.xlsx. A sample file is included in this repository. For proper operation of C_EANN_calculating.py, input data files must follow this format.
•	Output.xlsx - An Excel data file storing the predicted Chezy roughness coefficient results for the input data vectors specified in Input.xlsx. This file is generated as a result of executing C_EANN_calculating.py. To ensure proper execution, Output.xlsx must be closed if it was previously opened.

## Operational Principles

1.	To train the neural network ensemble, the training dataset file Training_Data.xlsx must be placed in the Data folder.
2.	To perform predictions using the ensemble, the input data file Input.xlsx must be prepared.
3.	Training of the neural network ensemble is performed using the C_EANN_training.py module.
4.	Training results are saved as separate text files containing weight matrices for each neural network: ANN_A, ANN_B1, and ANN_B2.
5.	Prediction based on the trained neural networks ANN_A, ANN_B1, and ANN_B2 is performed using the C_EANN_calculating.py module.
6.	Prediction results are saved in Output.xlsx.

## Author
Yaroslav Khodnevych,
ya.v.khodnevych@gmail.com

