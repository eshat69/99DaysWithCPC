import numpy as np

def Perceptron(input1, input2, weights, bias):
    outputP = input1 * weights[0] + input2 * weights[1] + bias * weights[2]

    if outputP > 0:
        outputP = 1
    else:
        outputP = 0

    return outputP

# Example usage and testing
weights = [0.5, 0.5, -1]  # Example weights
bias = -1  # Example bias

test_inputs = [
    (1, 1),
    (1, 0),
    (0, 1),
    (0, 0)
]

for x, y in test_inputs:
    outputP = Perceptron(x, y, weights, bias)
    print(f"{x} or {y} is : {outputP}")

