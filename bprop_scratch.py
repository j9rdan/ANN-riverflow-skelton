import numpy as np


class Neuron:

    def __init__(self, inputs, n_outputs):

        """
        Constructor to create a single neuron within a network's layer
        :param inputs:  (list) floats either directly from dataset (for input layer), or the output of a neuron in
        the previous layer (for hidden & output layers)
        """
        self.inputs = inputs
        self.weights = np.random.rand(n_outputs)  # (list) randomly initialised set of weights
        self.bias = np.random.rand()   # (float) initialise random bias
        self.output = 0.0
        self.delta = 0.0
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
        previous_weights = []
        for i, layer in enumerate(self.layers):
            if i != 0:
                # store weights of each neuron from previous layer:
                previous_weights = [self.layers[i-1][j].weights for j in range(len(self.layers[i-1]))]
                # print("layer", str(i-1), "weights:", previous_weights)
            for j, neuron in enumerate(layer):
                weights_in = [neuron_weights[j] for neuron_weights in previous_weights]    # store pr
                # print("incoming weights to layer", str(i), weights_in)
                weighted_sum = sum(np.multiply(neuron.inputs, weights_in)) + neuron.bias  # calc weighted sum (S)
                u = NeuralNetwork.sigmoid(weighted_sum)    # apply sigmoid to weighted sum (u=f(S))
                neuron.output = u   # save output into corresponding neuron
                new_inputs.append(u)   # store output into list

        print("final output:", new_inputs[-1])
        return new_inputs

    def back_prop(self, correct_output=1):

        deltas = []
        for i, layer in reversed(list(enumerate(self.layers))):    # work back from last layer
            for j, neuron in enumerate(layer):
                derivative = NeuralNetwork.sigmoid_dxdy(neuron.output)  # calc f'(S)
                delta = (correct_output - neuron.output) * derivative  # delta = error * f'(S)
                neuron.delta = delta
                # print("layer", str(j), delta)
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

    def update_weights(self, learn_rate):

        """
        Updates all weights and biases in the network by descending the gradient of the error.
        :param learn_rate:  (float) step size to adjust the gradient by (i.e. how quickly to learn)
        """
        for i in range(len(self.layers)):   # loop through layers
            if i < len(self.layers)-1:
                print("i:", str(i))
                # store delta of each neuron from next layer:
                deltas = [self.layers[i+1][j].delta for j in range(len(self.layers[i][0].weights))]
                # print("deltas:", deltas)
                for neuron in self.layers[i]:   # for each neuron in a layer
                    print("current weights:", neuron.weights)
                    neuron.weights = neuron.weights * deltas * learn_rate * neuron.output   # update weights
                    neuron.bias += learn_rate * neuron.delta    # update bias
                    # print("current bias:", neuron.bias)
                    print("new weights:", neuron.weights)
                    # print("new bias:", neuron.bias)


    def train(self, data, epochs, l_rate):
        pass




# n1 = Neuron([1], 3)
# n2 = Neuron([2], 3)
# n3 = Neuron([3, 3], 1)
# n4 = Neuron([2, 2], 1)
# n5 = Neuron([2, 2], 1)
# n6 = Neuron([2, 2, 2], 1)
#
#
# l1 = [n1, n2]
# l2 = [n3, n4, n5]
# l3 = [n6]
#
# mlp = NeuralNetwork([l1, l2, l3])
#
# for epoch in range(15):
#     mlp.fwrd_prop()
#     mlp.back_prop()
#     mlp.update_weights(0.5)


dataset = [[np.random.rand() for i in range(3)] for j in range(5)]

# get no. of hidden neurons
n_hidden = int(input("Enter no. of hidden neurons: "))

# create layers
input_layer = [Neuron([column], n_hidden) for column in dataset[0]]
hidden_layer = [Neuron([], 1) for k in range(n_hidden)]
output_layer = [Neuron([], 1)]

# create network
neural_network = NeuralNetwork([input_layer, hidden_layer, output_layer])









