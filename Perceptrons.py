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

