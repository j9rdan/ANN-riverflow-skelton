import csv
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
        # self.db = 0.0


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

        """
        Calculates all the weighted sums starting from the input layer, using each neuron's incoming weights and
        respective bias, before applying the sigmoid function to each result and storing it in each neuron.

        :return new_inputs:   (list) results from sigmoid function for all neurons
        """

        new_inputs = []
        previous_weights = []
        for i, layer in enumerate(self.layers):
            if i != 0:
                # store weights of each neuron from previous layer:
                previous_weights = [self.layers[i-1][j].weights for j in range(len(self.layers[i-1]))]
                # print("layer", str(i-1), "weights:", previous_weights)
            for j, neuron in enumerate(layer):
                # store incoming weights for each neuron:
                weights_in = [neuron_weights[j] for neuron_weights in previous_weights]
                # print("incoming weights to layer", str(i), weights_in)
                weighted_sum = sum(np.multiply(neuron.inputs, weights_in)) + neuron.bias  # calc weighted sum (S)
                u = NeuralNetwork.sigmoid(weighted_sum)    # apply sigmoid to weighted sum (u=f(S))
                neuron.output = u   # save output into corresponding neuron
                new_inputs.append(u)   # store output into list

        # print("outputs:", new_inputs)
        return new_inputs

    def back_prop(self, correct_output=1):

        """
        Calculates delta for each neuron using the derivative of its output from the sigmoid function, starting back
        from the output layer towards the input layer.

        :param correct_output:  (float) standardised value we are trying to predict
        :return deltas:         (list) resulting deltas to be used in updating the weights
        """

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

    def update_weights(self, learn_rate):

        """
        Updates all weights and biases in the network by descending the gradient of the error.
        :param learn_rate:  (float) step size to adjust the gradient by (i.e. how quickly to learn)
        """
        for i in range(len(self.layers)):   # loop through layers
            if i < len(self.layers)-1:
                # print("i:", str(i))
                # store delta of each neuron from next layer:
                deltas = [self.layers[i+1][j].delta for j in range(len(self.layers[i][0].weights))]
                # print("deltas:", deltas)
                for neuron in self.layers[i]:   # for each neuron in a layer
                    # print("current weights:", neuron.weights)
                    neuron.weights = neuron.weights * deltas * learn_rate * neuron.output   # update weights
                    neuron.bias += learn_rate * neuron.delta    # update bias
                    # print("current bias:", neuron.bias)
                    # print("new weights:", neuron.weights)
                    # print("new bias:", neuron.bias)

    @staticmethod
    def train(data, n_epochs, l_rate):

        """
        Creates a neural network with a given number of hidden neurons, which is then trained by forward propagating,
        back-propagating and updating the inputs and weights for each row. This is repeated (trained) for a given number
        of epochs

        :param data:    (2d list) dataset to train model on
        :param n_epochs: (int)    number of epochs to train for
        :param l_rate:  (float)   step size to adjust the gradient by (i.e. how quickly to learn)

        :return neural_network: (NeuralNetwork) trained network object
        """

        # get no. of hidden neurons
        n_hidden = int(input("Enter no. of hidden neurons: "))

        # create layers
        input_layer = [Neuron([column], n_hidden) for column in data[0]]  # use 1st row for input neurons
        hidden_layer = [Neuron([0] * len(input_layer), 1) for _ in range(n_hidden)]
        output_layer = [Neuron([0] * len(hidden_layer), 1)]

        # create network
        neural_network = NeuralNetwork([input_layer, hidden_layer, output_layer])

        for epoch in range(n_epochs):  # repeat for N epochs
            for i, row in enumerate(data):  # for every row
                neural_network.fwrd_prop()
                neural_network.back_prop(correct_output=1)  # correct value = last column in input data
                neural_network.update_weights(learn_rate=l_rate)
                if i != len(data) - 1:
                    new_inputs = data[i + 1]
                # print("new inputs:", new_inputs)
                for j, neuron in enumerate(input_layer):
                    neuron.inputs = [new_inputs[j]]
                    # print("neuron inputs:", neuron.inputs)
            print("epochs:", str(epoch + 1))

        return neural_network

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

    @staticmethod
    def tan_h(S):
        return (np.exp(S) - np.exp(-S)) / (np.exp(S) + np.exp(-S))

    @staticmethod
    def tan_h_dxdy(u):
        return 1 - (u ** 2)


####################################################################################################################
####################################################################################################################

# read & format csv file:
file = open('test_data.csv', 'r')
data_reader = csv.reader(file, delimiter=',')
dataset_str = [row[1:9] for row in data_reader]   # only number values from table
dataset_numstr = dataset_str[2:98]   # remove 98 when passing cleaned data
dataset = [[float(dataset_numstr[i][j]) for j in range(len(dataset_numstr[i]))]   # convert all str columns to float
           for i in range(len(dataset_numstr))]

# make a prediction:
trained_network = NeuralNetwork.train(dataset, n_epochs=1000, l_rate=0.5)
output = trained_network.fwrd_prop()[-1]   # get output layer output
print("Next predicted mean daily flow at Skelton:", output)







