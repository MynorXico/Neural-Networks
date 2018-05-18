""" Perceptrons"""
from numpy import dot
import math

def step_function(x):
    return 1 if x >= 0 else 0

def perceptron_output(weights, bias, x):
    """ returns 1 if the perceptrons 'fires', 0 if not """
    calculation = dot(weights, x) + bias;
    return step_function(calculation)

def sigmoid(t):
    return 1/(1+math.exp(-t))

def neuron_output(weights, inputs):
    return sigmoid(dot(weights, inputs))


def feed_forward(neural_network, input_vector):
    """ takes in a neural netwoek
    (represented as a list of lists of lists of weights
    and returns the output from forward-propagating the input"""

    outputs = []

    # process one layer at a time
    for layer in neural_network:
        input_with_bias = input_vector +[1]
        output = [neuron_output(neuron, input_with_bias)
                  for neuron in layer]
        outputs.append(output)

        # then the input to the next layer is the output of this one
        input_vector = output
    return outputs

xor_network = [#hidden layer
                [[20, 20, -30], # 'and' neuron
                 [20, 20, -10]], # 'or' neuron
                #output layer
                [[-60, 60, -30]]] # '2nd input but not 1st input' neuron
for x in[0, 1]:
    for y in [0, 1]:
        # feed_forward produces the outputs of every neuron
        # feed_forward[-1] is the outputs of the output-layer neurons
        print(str(x)+str(y)+str(feed_forward(xor_network,[x, y])[-1]))
