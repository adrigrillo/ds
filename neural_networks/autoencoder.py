# -*- coding: utf-8 -*-
"""
autoencoder.py
=================

Implementation of a neural network using stochastic gradient descent.
"""
from typing import List

import numpy as np
import numpy.random as random


class Autoencoder(object):

    def __init__(self, sizes: List[int] = None, learning_rate: float = 0.01, regularization_penalty: float = 0):
        """
        Initializes the neural network creating the layers of neurons, the matrix of weights, the biases and the
        required data structures for a neural network.
        This method will require an list of ints that indicates the size of each layer (the number of neurons). The
        learning rate (alpha) and the regularization penalty (lambda) could be set.

        :param sizes: array with the sizes of each layer. Default: [8, 3, 8]
        :param learning_rate: learning rate of the neural network (alpga). Default: 0.01
        :param regularization_penalty: regularization penalty (lambda). Default: 0
        """
        if sizes is None:
            sizes = [8, 3, 8]
        self.learning_rate = learning_rate  # alpha
        self.regularization_penalty = regularization_penalty  # lambda
        self.number_of_layers = len(sizes)
        self.sizes = sizes
        self.weighted_inputs = None  # z's
        self.activations = None  # a's
        self.deltas = None  # deltas
        self.biases = [random.rand(layer_size, 1) for layer_size in sizes[1:]]
        self.bias_partial_derivatives = [np.zeros(bias.shape) for bias in self.biases]
        self.weights = [random.rand(layer_size, prev_layer_size)
                        for layer_size, prev_layer_size in zip(sizes[1:], sizes[:-1])]
        self.weights_partial_derivatives = [np.zeros(weight.shape) for weight in self.weights]
        self.cost_gradients = [random.rand(layer_size, prev_layer_size)
                               for layer_size, prev_layer_size in zip(sizes[1:], sizes[:-1])]
        self.epoch_convergence = None
        self.errors = []

    def gradient_descent_learning(self, training_examples: np.ndarray, epochs: int = 50000, batch_size: int = 8,
                                  evaluation: bool = True) -> None:
        """
        Method that trains the neural network and evaluate its results for a given input.
        The batch size will generate a training input with the indicated number of instances that will be used to
        update the weights and bias of the neuron, if the batch is smaller than the total training set, this last will
        be divided in n batch that will be used to train the neural network.

        :param training_examples: matrix with the inputs to the network
        :param epochs: integer with the number of times the training will be done. Default: 10000 epochs
        :param batch_size: size of the training batch used per gradient descent update. Default: 8 samples
        :param evaluation: true to print the evaluation of the network after each epoch. Default: True
        :return:
        """
        num_training_examples = training_examples.shape[1]
        for i in range(int(epochs)):
            random.shuffle(np.transpose(training_examples))  # randomize the input data
            batches = [training_examples[:, i:i + batch_size] for i in range(0, num_training_examples, batch_size)]
            for batch in batches:
                self.train_neural_network(batch)
            if evaluation:
                error = np.sum(self.deltas[-1]) ** 2
                self.errors.append(error)
                correct_results = self.evaluate(training_examples)
                print('Epoch: {0}. Correct results: {1} of {2}'.format(i, correct_results, num_training_examples))
                if correct_results == num_training_examples:
                    if self.epoch_convergence is None:
                        self.epoch_convergence = i
                    print('Converged in {0} epochs'.format(self.epoch_convergence))
                    if error < 0.01:
                        print("Squared error less than 0.01 in {0} epochs. Value: {1}".format(i, error))
                        break

    def train_neural_network(self, input_values: np.ndarray):
        """
        Method that handle the training process for a set of inputs. This method create the structures to save the
        data during the training process and call the different methods that make the calculations.

        :param input_values: training values
        """
        input_length = input_values.shape[1]
        self.weighted_inputs = [np.zeros([layer_size, input_length]) for layer_size in self.sizes[1:]]
        self.activations = [np.zeros([layer_size, input_length]) for layer_size in self.sizes]
        self.deltas = [np.zeros([layer_size, input_length]) for layer_size in self.sizes]
        self.forward_propagation(input_values)
        self.backwards_propagation(input_values, input_length)
        self.gradient_descent()

    def forward_propagation(self, net_input: np.ndarray):
        """
        Forward propagation process.

        :param net_input: array with the input of the network
        :return: output of the network
        """
        self.activations[0] = net_input  # save the input as the activation of the first layer
        for i in range(len(self.weights)):
            weighted_input = np.dot(self.weights[i], net_input) + self.biases[i]  # numpy broadcasting
            net_input = sigmoid(weighted_input)
            self.weighted_inputs[i] = weighted_input  # saved for back propagation -> z
            self.activations[i + 1] = net_input  # saved for back propagation -> a

    def backwards_propagation(self, expected_outputs: np.ndarray, training_examples_length: int):
        """
        Backwards propagation process.

        :param expected_outputs: output of the last layer of the neural network should show
        :param training_examples_length: length of the training example contained in the input. Equal to m
        """
        training_examples_div = (1 / training_examples_length)  # 1/m
        self.deltas[-1] = -(expected_outputs - self.activations[-1])  # -(y - activation_output)
        # self.deltas[-1] = -(expected_outputs - self.activations[-1]) * sigmoid_derivative(self.activations[-i])
        # -(y - activation_output) * f'(z[-1])
        self.weights_partial_derivatives[-1] = \
            training_examples_div * np.dot(self.deltas[-1], np.transpose(self.activations[-2]))
        self.bias_partial_derivatives[-1] = training_examples_div * np.sum(self.deltas[-1], axis=1, keepdims=True)
        for i in range(2, self.number_of_layers):
            self.deltas[-i] = np.dot(np.transpose(self.weights[-i + 1]), self.deltas[-i + 1]) \
                              * sigmoid_derivative(self.activations[-i])
            self.weights_partial_derivatives[-i] = \
                training_examples_div * np.dot(self.deltas[-i], np.transpose(self.activations[-i - 1]))
            self.bias_partial_derivatives[-i] = training_examples_div * np.sum(self.deltas[-i], axis=1, keepdims=True)

    def gradient_descent(self):
        """
        Method that handle the gradient descent calculations, take care of the calculation required to update
        the weights and biases.
        This method depend of the learning rate set during the creation of the neural network ('learning-rate) and
        it is capable of doing regularization setting the value of 'regularization_penalty'.
        """
        for i in range(len(self.weights)):
            self.weights[i] -= \
                self.learning_rate * (
                        self.weights_partial_derivatives[i] + self.regularization_penalty * self.weights[i])
            self.biases[i] -= \
                self.learning_rate * (self.bias_partial_derivatives[i] + self.regularization_penalty * self.biases[i])

    def evaluate(self, data: np.ndarray):
        """
        Function that takes all the training data and check the results that the network returns.
        The one in the neural network output will be the position where the result is higher and then is compared
        with the input.

        :param data: data used to compare the results
        :return: number of correct results returned by the network
        """
        self.forward_propagation(data)
        return np.sum(np.argmax(self.activations[-1], axis=0) == np.argmax(data, axis=0))


def sigmoid(z):
    """
    The sigmoid function that applies to the result of the product of the weight and the activation of the
    neurons plus the biases, known as weighted input.

    z = w_l*a_l+b

    :param z: weighted input.
    :return: activation of the next layer of the network
    """
    return 1.0 / (1 + np.exp(-z))


def sigmoid_derivative(activation_values):
    """
    Derivative of the sigmoid function

    :param activation_values: activation values
    :return: result of applying the derivative function
    """
    return activation_values * (1.0 - activation_values)


if __name__ == '__main__':
    # Default training set
    array = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1]])

    # Creation of the network and training
    # network = nn.NeuralNetwork(learning_rate=0.1)
    # If the learning rate or the penalty wants to be changed
    network = Autoencoder(learning_rate=0.1, regularization_penalty=0.0001)
    network.gradient_descent_learning(array)

    # Testing
    test_input = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1]])

    print('Test input:')
    print(np.array_str(test_input))
    network.forward_propagation(test_input)
    print('Last layer activation:')
    print(np.array_str(network.activations[-1], precision=3, suppress_small=True))
    print('Positions where is located the one:')
    print('For the input: {0}'.format(np.argmax(test_input, axis=0)))
    print('For the output of the neural network: {0}'.format(np.argmax(network.activations[-1], axis=0)))
    print('Weights and bias of the network')
    print('Weights and bias from input to hidden layer:')
    print(np.array_str(network.weights[0], precision=3, suppress_small=True))
    print(np.array_str(network.biases[0], precision=3, suppress_small=True))
    print('Weights and bias from hidden layer to output:')
    print(np.array_str(network.weights[1], precision=3, suppress_small=True))
    print(np.array_str(network.biases[1], precision=3, suppress_small=True))
