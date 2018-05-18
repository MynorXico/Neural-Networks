""" Perceptrons"""
def dot(v, w):
    """v_1 * w_1 + ... + v_n * w_n"""
    return sum(v_i * w_i for v_i, w_i in zip(v, w))
import math
import random
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

def backpropagate(network, input_vector, targets):
    hidden_outputs, outputs = feed_forward(network, input_vector)

    # the output * (1-output) is from the derivative of sigmoid
    output_deltas = [output * (1-output) *(output-target)
                      for output, target in zip(outputs, targets)]

    #adjust weights for output layer, one neuron at a time
    for i, output_neuron in enumerate(network[-1]):
        # focus on the ith output layer neuron
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            #adjust the jth weight based on both
            # this neuron's delta and its jth input
            output_neuron[j] -= output_deltas[i]*hidden_output

    # back-propagate errors to hidden layer

    hidden_deltas = [hidden_output*(1-hidden_output)*
                         dot(output_deltas, [n[i] for n in output_layer])
                         for i, hidden_output in enumerate(hidden_outputs)]
        # adjust weights for hidden layer, one nueron at a time
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j]-=hidden_deltas[i] * input

""" building our neural network for captcha recognizer """
targets = [[1 if i == j else 0 for i in range(10)]
           for j in range(10)]

random.seed(0)  # to get repeatable results
input_size = 25 # each input is a vector of length 25
num_hidden = 5  # we'll have 5 neurons in the hidden layer
output_size = 10# we need 10 outputs for each input

# each hidden neuron has one wieght per input, plus a bias weight
hidden_layer = [[random.random() for _ in range(input_size +1 )
                 for _ in range(num_hidden)]]

# each output neuron has one weight per hidden neuron, plus a bias weight
output_layer = [[random.random() for _ in range(num_hidden +1)]
                for _ in range(output_size)]

# the network starts out with random weights
network = [hidden_layer, output_layer]


inputs = [[ 1,1,1,1,1,
            1,0,0,0,1,
            1,0,0,0,1,
            1,0,0,0,1,
            1,1,1,1,1],
          [ 0,0,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0],
          [ 1,1,1,1,1,
            0,0,0,0,1,
            1,1,1,1,1,
            1,0,0,0,0,
            1,1,1,1,1],
          [ 1,1,1,1,1,
            0,0,0,0,1,
            1,1,1,1,1,
            0,0,0,0,1,
            1,1,1,1,1],
          [ 1,0,0,0,1,
            1,0,0,0,1,
            1,1,1,1,1,
            0,0,0,0,1,
            0,0,0,0,1,],
          [ 1,1,1,1,1,
            1,0,0,0,0,
            1,1,1,1,1,
            0,0,0,0,1,
            1,1,1,1,1],
          [ 1,1,1,1,1,
            1,0,0,0,0,
            1,1,1,1,1,
            1,0,0,0,1,
            1,1,1,1,1],
          [ 1,1,1,1,1,
            0,0,0,0,1,
            0,0,0,0,1,
            0,0,0,0,1,
            0,0,0,0,1],
          [ 1,1,1,1,1,
            1,0,0,0,1,
            1,1,1,1,1,
            1,0,0,0,1,
            1,1,1,1,1],
          [ 1,1,1,1,1,
            1,0,0,0,1,
            1,1,1,1,1,
            0,0,0,0,1,
            1,1,1,1,1]]


""" Lets train our neural network"""
for _ in range(10000):
    for input_vector, target_vector in zip(inputs, targets):
        backpropagate(network, input_vector, target_vector)

print("Success!")

def predict(input):
    return feed_forward(network, input)[-1]

answer = predict([0,1,1,1,0, # .@@@.
                1,0,0,1,1, # @..@@
                0,1,1,1,0, # .@@@.
                1,0,0,1,1, # @..@@
                0,1,1,1,0]) # .@@@.

for x in range(len(answer)):
    answer[x] = round(answer[x]/max(answer),2)

print(answer)
