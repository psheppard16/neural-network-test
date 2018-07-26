import numpy as np
import copy

def sigmoid(x):
    return 1/(1+np.exp(-x))

def getHypothesis(X, synapse):
    return sigmoid(np.dot(X, synapse))

def getBinaryClassification(X, y, regularization, alpha=1.0, lamb=1, threshold=.00001, seed=1):
    if len(X[0]) != len(regularization):
        raise Exception("regularization list must have the same number of terms as features")
    if regularization[0] != 0:
        raise Exception("regularization[0] must equal 0")

    def costFunction(X, y, regularization, synapse):
        hypothesis = getHypothesis(X, synapse)
        cost = -y * np.log(hypothesis) - (1 - y) * np.log(1 - hypothesis)
        average = np.mean(cost)
        regularized = average + np.sum(regularization * regularization) * lamb / len(X[:,[0]])
        return regularized

    def getGradient(X, y, synapse):
        gradient = []
        for column in range(len(X[0])):
            hypothesis = getHypothesis(X, synapse)
            derivative = (hypothesis - y) * X[:,[column]]
            average = np.mean(derivative)
            regularized = average + regularization[column] / len(X[:,[column]])
            gradient.append([regularized])
        return np.array(gradient)

    np.random.seed(seed)
    synapse = 2*np.random.random((len(X[0]), 1)) - 1

    errorList = []
    while len(errorList) < 2 or errorList[-2] - errorList[-1] > threshold:
        delta = getGradient(X, y, synapse) * alpha
        synapse -= delta
        error = costFunction(X, y, regularization, synapse)
        errorList.append(error)
        if error > errorList[-1]:
            raise Exception("cost increased on last iteration: alpha may be too large")

    return synapse

def getMultiClassification(X, y, classes, alpha=1.0, threshold=.00001, seed=1):
    binaryYList = []
    for classification in classes: #set all values not equal to classification to 0 and all equal to 1
        binaryY = copy.deepcopy(y)
        binaryY[y != classification] = 0
        binaryY[y == classification] = 1
        binaryYList.append(binaryY)

    synapses = []
    for binaryY in binaryYList: #preform binary classification on all manipulated outcome sets
        synapse = getBinaryClassification(X, binaryY, np.array([0, 0, 0]), alpha=alpha, threshold=threshold, seed=seed)
        synapses.append(synapse)

    return synapses

def getClassification(case, synapses, classes):
    probabilities = []
    for synapse in synapses: #get the probability of each class
        probabilities.append(getHypothesis(case, synapse))

    highest = 0
    for probability in probabilities:
        if probability > highest:
            highest = probability

    index = probabilities.index(highest) #get index of most likely class
    return classes[index]

input = []
output = []
for x in range(-100, 100):
    for y in range(-100, 100):
        input.append([1, x / 100, y / 100])
        if x >= 0 and y >= 0:
            output.append([0])
        elif x < 0 and y > 0:
            output.append([1])
        elif x <= 0 and y <= 0:
            output.append([2])
        elif x > 0 and y < 0:
            output.append([3])

input = np.array(input)
output = np.array(output)

synapses = getMultiClassification(input, output, [0, 1, 2, 3], alpha=.1)
print(getClassification(np.array([1, 11, -1]), synapses, [0, 1, 2, 3]))

