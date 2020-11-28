# EE559-Deep Learning - Mini-Project 2

Design of a mini deep-learning framework (using only pytorch’s tensor operations and the standard math library).
Our framework provides the necessary tools to :

* build networks combining fully connected layers, ReLU, Leaky ReLU, ELU, Sigmoid and Tanh activations functions.
* run the forward and backward passes
* optimize parameters with SGD for MSE and Cross-Entropy Loss.


### Prerequisites

Make sure that ```Python >= 3.7``` and ```Pytorch >= 1.4``` are installed.

## Script files

### ```initializers.py```: Initializer class. Implements default, Xavier and He initialization.
### ```losses.py```: Loss class. Implements MSE and Cross-Entropy Loss.
### ```optimizer.py```: Optimizer class. Implements SGD algorithm.
### ```base.py```: Module class.
### ```trainable_layers.py```: Trainable Layer class. Implements the Linear layer (forward and backward pass).
### ```activations.py```: ActivationFunc class. Implements ReLU, Leaky ReLU, ELU, Sigmoid, Tanh and SoftMax functions (forward and backward pass).
### ```utils.py```: Utility functions to generate datasets, train, evaluate and compute errors.
### ```test.py```: Builds a network with two input units, two output units, three hidden layers of 25 units and trains it with MSE, logging the loss. Computes the final train and test errors.

## Run the project

~~~~shell
python test.py
~~~~

## Authors

* **Nicolas Jomeau**
* **Ella Rajaonson**
* **Sofia Dandjee**
