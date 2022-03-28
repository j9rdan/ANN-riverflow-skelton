import csv
import numpy as np
import math


class Neuron:

    def __init__(self, inputs, n_outputs):

        """
        Constructor to create a single neuron within a network's layer
        :param inputs:  (list) floats either directly from dataset (for input layer), or the output of a neuron in
        the previous layer (for hidden & output layers)
        """

        self.inputs = inputs
        self.weights = np.random.rand(n_outputs)  # (list) randomly initialised set of outgoing weights
        self.bias = np.random.rand()              # (float) initialise random bias
        self.output = 0.0
        self.delta = 0.0


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
        :return output:   (list) results from sigmoid function for all neurons
        """

        output = []
        previous_weights = []
        u = 0
        for i, layer in enumerate(self.layers):
            new_inputs = output
            output = []
            if i != 0:
                # store weights of each neuron from previous layer:
                previous_weights = [self.layers[i-1][j].weights for j in range(len(self.layers[i-1]))]
            for j, neuron in enumerate(layer):
                if new_inputs:
                    neuron.inputs = new_inputs
                # store incoming weights for each neuron:
                weights_in = [neuron_weights[j] for neuron_weights in previous_weights]
                weighted_sum = sum(np.multiply(neuron.inputs, weights_in)) + neuron.bias  # calc weighted sum (S)
                u = Calculator.sigmoid(weighted_sum)    # apply sigmoid to weighted sum (u=f(S))
                neuron.output = u   # save output into corresponding neuron
                output.append(u)   # store output into list
                # if i == len(self.layers) - 1:
                #     print("final output:", output)
            else:
                for j, neuron in enumerate(layer):   # update outputs of input layer
                    # print("layer", i)
                    # print("output", j, "old:", neuron.output)
                    neuron.output = neuron.inputs[0]
                    # print("output", j, "new:", neuron.output)

        # print("outputs:", output)
        return u

    def back_prop(self, correct_output):

        """
        Calculates delta for each neuron using the derivative of its output from the sigmoid function, starting back
        from the output layer towards the input layer.
        :param correct_output:  (float) standardised value we are trying to predict
        :return deltas:         (list) resulting deltas to be used in updating the weights
        """

        error = 0.0
        for i, layer in reversed(list(enumerate(self.layers))):    # work back from last layer
            for neuron in layer:
                derivative = Calculator.sigmoid_dxdy(neuron.output)  # calc f'(S)
                delta = (correct_output - neuron.output) * derivative  # delta = error * f'(S)
                neuron.delta = delta
                if i == 2:
                    error = neuron.output - correct_output

        return error

    def update_weights(self, learn_rate):

        """
        Updates all weights and biases in the network by descending the gradient of the error.
        :param learn_rate:  (float) step size to adjust the gradient by (i.e. how quickly to learn)
        """
        for i in range(len(self.layers)):   # loop through layers
            if i < len(self.layers)-1:
                # store delta of each neuron from next layer:
                deltas = [self.layers[i+1][j].delta for j in range(len(self.layers[i][0].weights))]
            for neuron in self.layers[i]:   # for each neuron in a layer
                if i < len(self.layers) - 1:
                    weight_change = [x * learn_rate for x in deltas]
                    weight_change = [x * neuron.output for x in weight_change]
                    neuron.weights += weight_change
                neuron.bias += learn_rate * neuron.delta  # update bias using wi,j = wi,j + p*δj*ui

    def train(self, data, n_epochs, l_rate):

        """
        Trains a neural network by forward propagating, back-propagating and updating the inputs and weights for each
        row. This is repeated for a given number of epochs
        :param data:     (2d list) dataset to train model on
        :param n_epochs: (int)     number of epochs to train for
        :param l_rate:   (float)   step size to adjust the gradient by (i.e. how quickly to learn)
        """

        for epoch in range(n_epochs):  # repeat for n epochs
            for i, row in enumerate(data):  # for every row
                prediction = self.fwrd_prop()
                error = self.back_prop(correct_output=row[4])  # correct value = mean daily flow at Skelton on next day
                if i == len(data)-1:    # output prediction on final row
                    print("prediction:", prediction)
                    print("error:", error)
                self.update_weights(learn_rate=l_rate)
                if i != len(data) - 1:
                    new_inputs = data[i + 1]
                for j, neuron in enumerate(self.layers[0]):
                    neuron.inputs = [new_inputs[j]]
                self.layers[-1][0].output = new_inputs[-1]  # set output of output neuron to correct value?
            print("epochs:", str(epoch + 1))


class Calculator:

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

        """
        Calculates the tanh activation function for a given weighted sum, S
        :param S:   (float) value of the weighted sum
        :return:    (float) output value, u = f(S)
        """
        return (np.exp(S) - np.exp(-S)) / (np.exp(S) + np.exp(-S))

    @staticmethod
    def tan_h_dxdy(u):

        """
        Calculates the derivative of the tanh activation function for a given activation, u
        :param u:   (float) value of the activation (i.e. f(S): tanh function applied to weighted sum)
        :return:    (float) derivative output, f'(S)
        """
        return 1 - (u ** 2)

    @staticmethod
    def RMSE(correct, actual):

        """
        Calculates the root mean squared error.
        :param correct: (float) actual value the model is aiming for
        :param actual:  (float) output value from training
        :return:        (float) value of root mean squared error
        """
        return math.sqrt(np.square(np.subtract(correct, actual)).mean())

    @staticmethod
    def destandardise(u):
        pass


####################################################################################################################
####################################################################################################################

# read & format csv file:
file = open('data.csv', 'r')
data_reader = csv.reader(file, delimiter=',')
dataset_str = [row[0:5] for row in data_reader]  # only number values from table
dataset = [[float(dataset_str[i][j]) for j in range(len(dataset_str[i]))]  # convert all str columns to float
           for i in range(len(dataset_str))]

# get no. of hidden neurons
n_hidden = int(input("Enter no. of hidden neurons: "))

# create layers
input_layer = [Neuron([dataset[0][i]], n_hidden) for i in range(len(dataset[0])-1)]  # use 1st row of data for input neurons
hidden_layer = [Neuron([0] * len(input_layer), 1) for _ in range(n_hidden)]
output_layer = [Neuron([0] * len(hidden_layer), 1)]

# create a network:
neural_network = NeuralNetwork([input_layer, hidden_layer, output_layer])
neural_network.train(dataset, n_epochs=1000, l_rate=0.5)

# make a prediction:
# prediction = neural_network.fwrd_prop()    # get output from output layer
# print("Next predicted mean daily flow at Skelton:", prediction)

