"""3-layer ANN"""

import scipy.special
import numpy

class NeuralNetwork:
    def __init__(self, input_nodes, hidden1_nodes, hidden2_nodes, output_nodes, learning_rate):
        # Set the no. of nodes in each layer and the learning rate
        self.i_nodes = input_nodes
        self.h1_nodes = hidden1_nodes
        self.h2_nodes = hidden2_nodes
        self.o_nodes = output_nodes
        self.lr = learning_rate

        # Weight matrices
        self.wih1 = numpy.random.normal(0.0, pow(self.h1_nodes, -0.5), (self.h1_nodes, self.i_nodes))
        self.wh1h2 = numpy.random.normal(0.0, pow(self.h2_nodes, -0.5), (self.h2_nodes, self.h1_nodes))
        self.who = numpy.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h2_nodes))

        # Set the activation function, the logistic sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        # Convert inputs into 2D arrays
        inputs_array = numpy.array(inputs_list, ndmin=2).T
        targets_array = numpy.array(targets_list, ndmin=2).T

        # Calculate signals into first hidden layer
        hidden1_inputs = numpy.dot(self.wih1, inputs_array)

        # Calculate the signals emerging from first hidden layer
        hidden1_outputs = self.activation_function(hidden1_inputs)

        # Calculate signals into second hidden layer
        hidden2_inputs = numpy.dot(self.wh1h2, hidden1_outputs)

        # Calculate the signals emerging from second hidden layer
        hidden2_outputs = self.activation_function(hidden2_inputs)

        # Calculate signals into output layer
        final_inputs = numpy.dot(self.who, hidden2_outputs)

        # Calculate the signals emerging from output layer
        final_outputs = self.activation_function(final_inputs)

        # Current error is (target - actual)
        output_errors = targets_array - final_outputs

        # Hidden2 layer errors are the output errors, split by the weights, recombined at hidden2 nodes
        hidden2_errors = numpy.dot(self.who.T, output_errors)

        # Hidden1 layer errors are the hidden2 errors, split by the weights, recombined at hidden1 nodes
        hidden1_errors = numpy.dot(self.wh1h2.T, hidden2_errors)

        # Update the weights for the links between the input and hidden1 layers
        self.wih1 += self.lr * numpy.dot((hidden1_errors * hidden1_outputs * (1.0 - hidden1_outputs)), numpy.transpose(inputs_array))
                                          
        # Update the weights for the links between the hidden1 and hidden2 layers
        self.wh1h2 += self.lr * numpy.dot((hidden2_errors * hidden2_outputs * (1.0 - hidden2_outputs)), numpy.transpose(hidden1_outputs))
                                           
        # Update the weights for the links between the hidden2 and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden2_outputs))
                                        
    def query(self, inputs_list):
        # Convert the inputs list into a 2D array
        inputs_array = numpy.array(inputs_list, ndmin=2).T

        # Calculate signals into first hidden layer
        hidden1_inputs = numpy.dot(self.wih1, inputs_array)

        # Calculate outputs from the first hidden layer
        hidden1_outputs = self.activation_function(hidden1_inputs)

        # Calculate signals into second hidden layer
        hidden2_inputs = numpy.dot(self.wh1h2, hidden1_outputs)

        # Calculate outputs from the second hidden layer
        hidden2_outputs = self.activation_function(hidden2_inputs)

        # Calculate signals into output layer
        final_inputs = numpy.dot(self.who, hidden2_outputs)

        # Calculate outputs from the output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
