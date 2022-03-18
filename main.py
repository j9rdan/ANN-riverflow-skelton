# from random import random
import numpy as np
import math


class NeuralNetwork:

    def __init__(self, layers):

        """
        Constructor requiring a list of integers representing the number of neurons in each layer.

        :param layers (list): [in, ..., out]
            in = number of inputs
            ... = number of neurons in hidden layer, >1 intermediary number represents multiple hidden layers
            out = number of outputs
        """

        self.layers = layers

        # store random weights & 0 for derivatives of weights
        # 2d array: rows = layers[i] (current layer neurons), columns = layers[i+1] (next layer neurons)
        self.weights = [np.random.rand(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        self.dw = [np.zeros((layers[i], layers[i + 1])) for i in range(len(layers) - 1)]

        # initialise all neuron outputs (activations) to 0
        # 2d array: list storing list of n 0's for each layer
        self.outputs = [np.zeros(layers[i]) for i in range(len(layers))]

        # store random biases & 0 for derivatives of biases
        # 2d array: list storing list of biases for each layer
        self.biases = [np.random.randn(layers[i+1], 1) for i in range(len(layers) - 1)]
        self.db = [np.zeros((layers[i+1], 1)) for i in range(len(layers) - 1)]

    def standardise(self, dataset):
        # return 0.8 * ((val - min)/(max - min)) + 0.1
        pass

    def destandardise(self, dataset):
        # return (standard - 0.1 / 0.8)(max - min) + min
        pass

    def sigmoid(self, x):
        # return output = 1.0 / (1 + np.exp(-x))
        pass

    def sigmoid_dxdy(self, x):
        # return x * (1.0 - x)
        pass

    def fwrd_prop(self, inputs):
        # neurons (activation) = inputs
        # iterate through network layers:
            # matrix mult between previous neurons & weights
            # apply sigmoid function
            # store neurons for backprop
        pass

    def back_prop(self, inputs):
        # iterate backwards through previous layer
            # apply sigmoid derivative function
            # get neurons for current layer
            # store derivatives after matrix mult
        pass

    def update_weights(self, learning_param):
        # loop through all weights:
            # weights = derivatives * learning_param
        pass

    def RMSE(self, modelled, observed):
        # return math.sqrt(np.square(np.subtract(modelled, observed)).mean())
        pass

    def train(self, inputs, targets, epochs, learning_param):
        # for i in len(epochs):
            # standardise()
            # iterate through all training data
            # fwrd_prop()
            # calc error
            # back_prop error
            # update_weights()
            # RMSE(), append to sum of errors
            # de-standardise()
        pass


if __name__ == "__main__":
    # import data
    # create ANN
    # train ANN
    # initialise inputs, targets
    # predict
    nn = NeuralNetwork([2, 5, 1])

    for val in enumerate(nn.db):
        print(val)
