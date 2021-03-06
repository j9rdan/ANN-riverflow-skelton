import csv
import numpy as np
import matplotlib.pyplot as plt


class Neuron:

    def __init__(self, inputs, n_outputs):

        """
        Constructor to create a single neuron within a network's layer
        :param inputs:   (list) floats either directly from dataset (for input layer), or the output of a neuron in
                         the previous layer (for hidden & output layers)
        :param n_outputs (int)  number of outgoing weights
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
        :return u:   (float) final result from sigmoid function for output layer neuron
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
            else:
                for j, neuron in enumerate(layer):   # update outputs of input layer
                    output.append(neuron.inputs[0])

        return u

    def back_prop(self, correct_output):

        """
        Calculates delta for each neuron using the derivative of its output from the sigmoid function, starting back
        from the output layer towards the input layer.
        :param correct_output:  (float) standardised value we are trying to predict
        :return error * error:  (float) square of the error to be used when calculating the root mean squared error
        """

        error = 0.0
        for i, layer in reversed(list(enumerate(self.layers))):    # work back from last layer
            for neuron in layer:
                derivative = Calculator.sigmoid_dxdy(neuron.output)  # calc f'(S)
                delta = (correct_output - neuron.output) * derivative  # delta = error * f'(S)
                neuron.delta = delta
                if i == 2:
                    error = correct_output - neuron.output  # get error of final output node

        return error * error

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
                    new_weights = [x * learn_rate for x in deltas]
                    new_weights = [x * neuron.output for x in new_weights] # update weights: w_i,j = w_i,j + (lrate * ??_j * u_i)
                    neuron.weights += new_weights
                neuron.bias += learn_rate * neuron.delta  # update bias: w_i,j = w_i,j + (lrate * ??_j)

    def train(self, data, n_epochs, l_rate):

        """
        Trains a neural network by forward propagating, back-propagating and updating the inputs and weights for each
        row. This is repeated for a given number of epochs
        :param data:        (2d list) dataset to train model on
        :param n_epochs:    (int)     number of epochs to train for
        :param l_rate:      (float)   step size to adjust the gradient by (i.e. how quickly to learn)
        :return errors_RMSE (list)    list of root mean squared errors
        """

        errors_RMSE = []
        for epoch in range(n_epochs):  # repeat for n epochs
            errors = []
            for i, row in enumerate(data):  # for every row
                self.fwrd_prop()
                error = self.back_prop(correct_output=row[4])  # correct value = mean daily flow at Skelton on next day
                errors.append(error)
                if i == len(data)-1:    # output prediction on final row
                    rmse_result = Calculator.RMSE(errors)
                    errors_RMSE.append(rmse_result)
                self.update_weights(learn_rate=l_rate)  # gradient descent
                if i != len(data) - 1:
                    new_inputs = data[i + 1]    # get next row of data
                for j, neuron in enumerate(self.layers[0]):
                    neuron.inputs = [new_inputs[j]]     # set input layer inputs to next row

        return errors_RMSE

    def predict(self, dataset):

        """
        Performs a single forward pass for all rows in the data to generate a prediction. Plots the results on a graph
        of the expected output against the actual output
        :param dataset: (list)  dataset to make prediction on
        """
        predictions = []
        correct_values = []
        for i, row in enumerate(dataset):
            correct_values.append(row[4])   # store all correct values
            predictions.append(self.fwrd_prop())    # store all predictions
            if i != len(dataset) - 1:
                new_inputs = dataset[i + 1]    # get next row of data
            for j, perceptron in enumerate(self.layers[0]):
                perceptron.inputs = [new_inputs[j]] # set input layer inputs to next row

        # plot graphs:
        plt.plot(predictions)
        plt.plot(correct_values)
        plt.legend(["Predictions", "Expected"])
        plt.title("Skelton Mean Daily Flow: Actual vs Predicted")
        plt.show()


class Calculator:

    """
    Class responsible for math calculations required in the forward and backward pass.
    """

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
    def RMSE(errors):

        """
        Calculates the root mean squared error.
        :param errors:  (list) list of errors for final output node
        :return:        (float) value of root mean squared error
        """
        return np.sqrt(sum(errors) / len(errors))


####################################################################################################################
####################################################################################################################

# train data
train_file = open('train.csv', 'r')
train_reader = csv.reader(train_file, delimiter=',')
train_str = [row[0:5] for row in train_reader]  # only number values from table
train_data = [[float(train_str[i][j]) for j in range(len(train_str[i]))]  # convert all str columns to float
              for i in range(len(train_str))]

# test data
test_file = open('test.csv', 'r')
test_reader = csv.reader(test_file, delimiter=',')
test_str = [row[0:5] for row in test_reader]  # only number values from table
test_data = [[float(test_str[i][j]) for j in range(len(test_str[i]))]  # convert all str columns to float
             for i in range(len(test_str))]


# validation data
validation_file = open('validation.csv', 'r')
validation_reader = csv.reader(validation_file, delimiter=',')
validation_str = [row[0:5] for row in validation_reader]
validation_data = [[float(validation_str[i][j]) for j in range(len(validation_str[i]))]
                   for i in range(len(validation_str))]


# get no. of hidden neurons
n_hidden = int(input("Enter no. of hidden neurons: "))

# create layers
input_layer = [Neuron([train_data[0][i]], n_hidden) for i in range(len(train_data[0]) - 1)]  # use 1st row of data for input neurons
hidden_layer = [Neuron([0] * len(input_layer), 1) for _ in range(n_hidden)]
output_layer = [Neuron([0] * len(hidden_layer), 1)]

# create a network:
neural_network = NeuralNetwork([input_layer, hidden_layer, output_layer])
epochs = 10000

# train using training and test sets:
train_errors = neural_network.train(train_data, n_epochs=epochs, l_rate=0.5)
test_errors = neural_network.train(test_data, n_epochs=epochs, l_rate=0.5)

# plot error graphs:
n_epochs = [i for i in range(epochs)]
plt.plot(n_epochs, train_errors)
plt.title("RMSE for Training Data")
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.show()

plt.plot(n_epochs, test_errors)
plt.title("RMSE for Test Data")
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.show()

# predict using validation set:
neural_network.predict(validation_data)

