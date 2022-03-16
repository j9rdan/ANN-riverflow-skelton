from random import random
import numpy as np
import math


class NeuralNetwork:

    def __init__(self, num_inputs, num_h_layers, num_outputs):
        # initialise params
        # create layer structure
        # set random weights
        # store neurons per layer
        # store derivatives per layer
        pass

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
    pass