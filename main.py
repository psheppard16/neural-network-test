from layer import Layer
import layer as network
import numpy as np

def getClassification(hypothesis, threshold=.5):
    highIndexes = []
    for index in range(len(hypothesis[0])):
        if hypothesis[0][index] > threshold:
            highIndexes.append(index)

    result = [0 for x in range(len(hypothesis[0]))]
    for index in highIndexes:
        result[index] = 1
    return result

input = []
output = []
for x in range(-100, 100):
    for y in range(-100, 100):
        input.append([1, x / 100, y / 100])
        if x > 0 and y > 0:
            output.append([1, 0, 0, 0])
        elif x < 0 and y > 0:
            output.append([0, 1, 0, 0])
        elif x < 0 and y < 0:
            output.append([0, 0, 1, 0])
        elif x > 0 and y < 0:
            output.append([0, 0, 0, 1])
        elif x == 0 and y > 0:
            output.append([1, 1, 0, 0])
        elif x == 0 and y < 0:
            output.append([0, 0, 1, 1])
        elif y == 0 and x > 0:
            output.append([1, 0, 0, 1])
        elif y == 0 and x < 0:
            output.append([0, 1, 1, 0])
        elif x == 0 and y == 0:
            output.append([0, 0, 0, 0])



input = np.array(input)
output = np.array(output)

layer1 = Layer(3, None, 10, input=input, inputLayer=True)
layer2 = Layer(3, layer1, .1)
layer3 = Layer(3, layer2, .001)
layer4 = Layer(4, layer3, None, outputLayer=True)

network.train(output, iterations=90000)

testCase = np.array([[1, 100, 100]])
print(getClassification(network.getHypothesis(testCase)))
testCase = np.array([[1, -100, 100]])
print(getClassification(network.getHypothesis(testCase)))
testCase = np.array([[1, -100, -100]])
print(getClassification(network.getHypothesis(testCase)))
testCase = np.array([[1, 100, -100]])
print(getClassification(network.getHypothesis(testCase)))

print("   ")

testCase = np.array([[1, 0, 100]])
print(getClassification(network.getHypothesis(testCase)))
testCase = np.array([[1, 0, -100]])
print(getClassification(network.getHypothesis(testCase)))
testCase = np.array([[1, 100, 0]])
print(getClassification(network.getHypothesis(testCase)))
testCase = np.array([[1, -100, 0]])
print(getClassification(network.getHypothesis(testCase)))

print("   ")

testCase = np.array([[1, 0, 0]])
print(getClassification(network.getHypothesis(testCase)))