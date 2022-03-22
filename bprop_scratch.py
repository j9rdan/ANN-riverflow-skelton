import numpy as np


class Neuron:

    def __init__(self, inputs):

        """
        Constructor to create a single neuron within a network's layer
        :param inputs:  (list) floats either directly from dataset (for input layer), or the output of a neuron in
        the previous layer (for hidden & output layers)
        """
        self.inputs = inputs
        self.weights = np.random.rand(len(inputs))  # (list) randomly initialised set of weights
        self.bias = np.random.rand()   # (float) initialise random bias
        self.output = 0.0
        self.dw = []
        self.db = 0.0


class NeuralNetwork:

    def __init__(self, layers):

        """
        Constructor to create a neural network by defining a general structure for the input, hidden & output layer
        :param layers:   (2d list) [layer1, layer2, layer3]
            layer1 = [i1, i2, ..., i_j] : input layer with j input neurons
            layer2 = [h1, h2, ..., h_k] : hidden layer with k hidden neurons
            layer3 = [o1] : output layer with 1 output neuron
        """
        self.layers = layers    # list of (list of neuron objects)

    def fwrd_prop(self):

        new_inputs = []
        for layer in self.layers:
            print()
            for neuron in layer:
                weighted_sum = np.dot(neuron.inputs, neuron.weights) + neuron.bias  # calc weighted sum (S)
                u = NeuralNetwork.sigmoid(weighted_sum)    # apply sigmoid to weighted sum (u=f(S))
                neuron.output = u   # save output into corresponding neuron
                new_inputs.append(u)   # store output into list

        # print(new_inputs)
        return new_inputs

    def back_prop(self, correct_output=1):

        deltas = []
        for i, layer in reversed(list(enumerate(self.layers))):    # start from last layer
            for neuron in layer:
                derivative = NeuralNetwork.sigmoid_dxdy(neuron.output)  # calc f'(S)
                # neuron.dw.append(derivative) ??
                delta = (correct_output - neuron.output) * derivative  # delta = error * f'(S)
                print(delta)
                deltas.append(delta)

        # print(deltas)
        return deltas


    @staticmethod
    def sigmoid(S):

        """
        Calculates the sigmoid activation function for a given weighted sum, S
        :param S:   (float) value of the weighted sum
        :return:    (float) output value, u = f(S)
        """
        return 1.0 / (1.0 + np.exp(-S))

    @staticmethod
    def sigmoid_dxdy(u):

        """
        Calculates the derivative of the sigmoid activation function for a given activation, u
        :param u:   (float) value of the activation (i.e. f(S): sigmoid function applied to weighted sum)
        :return:    (float) derivative output, f'(S)
        """
        return u * (1.0 - u)



p1 = Neuron([1, 1, 1, 1, 1])
p2 = Neuron([2, 2, 2, 2, 2])
p3 = Neuron([3, 3, 3, 3, 3])

l1 = [p1, p2]
l2 = [p1, p2, p3]
l3 = [p1]

mlp = NeuralNetwork([l1, l2, l3])

mlp.fwrd_prop()
mlp.back_prop()