import numpy as np
import random
class Layer:
    layers = []
    times = []
    def __init__(self, nodes, previousLayer, alpha, input=None, inputLayer=False, outputLayer=False, seed=1):
        if not Layer.layers and not inputLayer:
            raise Exception("The first layer must be inputLayer=True")
        if inputLayer and not input.any():
            raise Exception("The first layer must be sent test cases")
        if inputLayer and outputLayer:
            raise Exception("A layer cannot be both the input and output layer")
        if outputLayer and alpha:
            raise Exception("The output layer cannot have a value for alpha, as it doesnt train")
        if alpha and alpha < 0:
            raise Exception("Alpha must be greater than 0")

        self.inputLayer = inputLayer
        self.outputLayer = outputLayer
        self.delta = None
        self.nextLayer = None
        self.activation = input
        self.alpha = alpha
        self.nodes = nodes
        self.previousLayer = previousLayer

        self.synapse = None
        if previousLayer:
            Layer.layers[-1].nextLayer = self

            np.random.seed(seed)
            self.previousLayer.synapse = 2*np.random.random((self.previousLayer.nodes, self.nodes)) - 1


        Layer.layers.append(self)

    def setActivation(self):
        if self.inputLayer:
            raise Exception("cannot set activations for input layer")
        if not self.previousLayer.activation.any():
            raise Exception("A value for activation in the previous layer has yet to be calculated")
        self.activation = sigmoid(np.dot(self.previousLayer.activation, self.previousLayer.synapse))
        if not self.outputLayer: #sets the bias units to be one
            self.activation[:,0] = 1

    def setDelta(self):
        if not self.activation.any():
            raise Exception("A value for activation has yet to be calculated")
        if not self.nextLayer:
            raise Exception("There is no next layer")
        if not self.nextLayer.delta.any():
            raise Exception("A value for delta in the next layer has yet to be calculated")
        self.delta = np.dot(self.nextLayer.delta, self.synapse.T) * sigmoid(self.activation, deriv=True)

    def updateWeights(self):
        if not self.synapse.any():
            raise Exception("Values for synapse have yet to be initiated")
        if self.outputLayer:
            raise Exception("The output layer has no synapses")
        if not self.alpha:
            raise Exception("Cannot update weights with no value for alpha")
        self.synapse += self.alpha * np.dot(self.activation.T, self.nextLayer.delta) / len(self.activation) / len(Layer.layers[-1].activation[0])

def train(output, threshold=None, iterations=None, debug=False):
    if not threshold and not iterations:
        raise Exception("A value for either threshold or iterations must be given")
    if threshold and iterations:
        raise Exception("A value for only threshold or iterations may be given")
    if not Layer.layers[0].inputLayer:
        raise Exception("The first layer must be inputLayer=True")
    if not Layer.layers[-1].outputLayer:
        raise Exception("The last layer must be outputLayer=True")

    if threshold:
        error1 = 10000
        error2 = 1000000000
        while error2 - error1 > threshold:
            print(error2 - error1, threshold)
            getForwardPropagation()
            backPropagate(output, debug)
            error2 = error1
            error1 = costFunction(output)
            if error1 > error2:
                raise Exception("cost increased on last iteration: alpha may be too large")
    elif iterations:
        errorList = []
        while iterations > 0:
            iterations -= 1
            getForwardPropagation()
            backPropagate(output, debug)
            error = costFunction(output)
            errorList.append(error)
            if len(errorList) > 1 and errorList[-1] > errorList[-2]:
                raise Exception("cost increased on last iteration: alpha may be too large")

def getHypothesis(testCase):
    hypothesis = testCase
    for layer in Layer.layers:
        if not layer.outputLayer:
            hypothesis = sigmoid(np.dot(hypothesis, layer.synapse))
    return hypothesis

def getForwardPropagation():
    if not Layer.layers[0].inputLayer:
        raise Exception("The first layer must be inputLayer=True")
    if not Layer.layers[-1].outputLayer:
        raise Exception("The last layer must be outputLayer=True")
    for layer in Layer.layers:
        if layer.previousLayer:
            layer.setActivation()
    return Layer.layers[-1].activation

def backPropagate(y, approximate):
    if not Layer.layers[0].inputLayer:
        raise Exception("The first layer must be inputLayer=True")
    if not Layer.layers[-1].outputLayer:
        raise Exception("The last layer must be outputLayer=True")

    outputLayer = Layer.layers[-1]
    outputLayer.delta = y - outputLayer.activation
    nextLayer = outputLayer.previousLayer
    while not nextLayer.inputLayer: #backpropagates until the input layer is reached, then stops
        nextLayer.setDelta()
        if approximate:
            delta = [[0 for x in range(len(nextLayer.synapse[0]))] for y in range(len(nextLayer.synapse))]
            for m in range(len(nextLayer.synapse[0])):
                for n in range(len(nextLayer.synapse)):
                    delta[n][m] = approxGradient(nextLayer, n, m, y)
            print(delta - np.dot(nextLayer.activation.T, nextLayer.nextLayer.delta) / len(nextLayer.activation) / len(Layer.layers[-1].activation[0]))
        nextLayer = nextLayer.previousLayer

    for layer in Layer.layers:
        if not layer.outputLayer:
            layer.updateWeights()

def costFunction(y):
    hypothesis = getForwardPropagation()
    cost = -y * np.log(hypothesis) - (1 - y) * np.log(1 - hypothesis)
    average = np.mean(cost)
    return average

def approxGradient(layer, n, m, y, epsilon=.00000001):
    layer.synapse[n][m] += epsilon
    plusCost = costFunction(y)
    layer.synapse[n][m] -= 2 * epsilon
    minusCost = costFunction(y)
    approx = (plusCost - minusCost) / 2 / epsilon
    layer.synapse[n][m] += epsilon
    return -approx

def sigmoid(x, deriv=False):
    if deriv:
        return x*(1-x) #returns the derivitive of the sigmoid function
    return 1/(1+np.exp(-x)) #returns the sigmoid function
