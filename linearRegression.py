import numpy as np
from numpy import linalg

def getNormalRegression(X, y):
    return np.dot(np.dot(linalg.pinv(X.T.dot(X)), X.T), y)

def getGradientRegression(X, y, alpha=1, threshold=.00001):
    def getHypothesis(X, synapse):
        return np.dot(X, synapse)

    def costFunction(X, y, synapse):
        hypothesis = getHypothesis(X, synapse)
        difference = hypothesis - y
        square = np.square(difference)
        average = np.mean(square)
        return average

    def regression(X, y, synapse, column):
        hypothesis = getHypothesis(X, synapse)
        deriv = (hypothesis - y) * X[:,[column]]
        average = np.mean(deriv)
        return average

    np.random.seed(1)
    synapse = 2*np.random.random((len(X[0]), 1)) - 1

    errorList = []
    while len(errorList) < 2 or errorList[-2] - errorList[-1] > threshold:
        delta = []
        for column in range(len(X[0])):
            delta.append([regression(X, y, synapse, column) * alpha])
        delta = np.array(delta)
        synapse -= delta
        error = costFunction(X, y, synapse)
        errorList.append(error)
        if error > errorList[-1]:
            raise Exception("cost increased on last iteration: alpha may be too large")

    return synapse

input = []
output = []

#for x in range(-100, 100):
#    for y in range(-100, 100):
#        input.append([1, x / 100, y / 100])
#        output.append([5 + x / 100 * 10 + y / 100])

for x in range(-100, 100):
    for y in range(-100, 100):
        input.append([1, x])
        output.append([y])

input = np.array(input)
output = np.array(output)

print(getNormalRegression(input, output))


