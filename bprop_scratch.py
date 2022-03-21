import numpy as np


class Neuron:

    def __init__(self, inputs):
        self.inputs = inputs
        self.weights = np.random.rand(len(inputs))
        self.bias = np.random.rand()
        self.output = 0.0
        self.dw = []
        self.db = 0.0


class NeuralNetwork:

    def __init__(self, layers):
        self.layers = layers

    def fwrd_prop(self):

        new_inputs = []
        for layer in self.layers:
            print()
            for neuron in layer:
                weighted_sum = np.dot(neuron.inputs, neuron.weights) + neuron.bias
                u = NeuralNetwork.sigmoid(weighted_sum)
                new_inputs.append(u)

        # print(new_inputs)
        return new_inputs

    def back_prop(self):
        

    @staticmethod
    def sigmoid(S):

        """
        Calculates the sigmoid activation function for a given weighted sum, S
        :param S:   (float) value of the weighted sum
        :return:    (float) output value, u = f(S)
        """
        return 1.0 / (1.0 + np.exp(-S))



p1 = Neuron([1, 1, 1, 1, 1])
p2 = Neuron([2, 2, 2, 2, 2])
p3 = Neuron([3, 3, 3, 3, 3])

l1 = [p1, p2]
l2 = [p1, p2, p3]
l3 = [p1]

mlp = NeuralNetwork([l1, l2, l3])

mlp.fwrd_prop()