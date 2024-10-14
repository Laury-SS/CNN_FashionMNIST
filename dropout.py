import numpy as np
from layer import Layer


class Dropout(Layer):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate
        self.mask = None

    def forward(self, input, training=True):
        self.input = input
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=input.shape) / (1 - self.rate)
            return input * self.mask
        return input

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.mask
