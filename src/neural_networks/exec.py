import numpy as np

from src.neural_networks.autoencoder import Autoencoder

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