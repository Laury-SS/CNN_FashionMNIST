import numpy as np
from layer import Layer


class MaxPooling(Layer):
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, input):
        self.input = input
        batch_size, height, width = input.shape

        pool_height, pool_width = self.pool_size
        out_height = height // pool_height
        out_width = width // pool_width

        output = np.zeros((batch_size, out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                region = input[:, i * pool_height:(i + 1) * pool_height, j * pool_width:(j + 1) * pool_width]
                output[:, i, j] = np.max(region, axis=(1, 2))

        return output

    def backward(self, output_gradient, learning_rate):
        batch_size, height, width = self.input.shape

        pool_height, pool_width = self.pool_size
        out_height = height // pool_height
        out_width = width // pool_width

        input_gradient = np.zeros_like(self.input)

        for i in range(out_height):
            for j in range(out_width):
                region = self.input[:, i * pool_height:(i + 1) * pool_height, j * pool_width:(j + 1) * pool_width]
                max_region = np.max(region, axis=(1, 2), keepdims=True)
                mask = (region == max_region)
                input_gradient[:, i * pool_height:(i + 1) * pool_height,
                j * pool_width:(j + 1) * pool_width] += mask * output_gradient[:, i, j][:, None, None]

        return input_gradient
