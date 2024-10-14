import time

import numpy as np

from dropout import Dropout
from misc import preprocess_data
from utils import mnist_reader

from max_pooling import MaxPooling
from dense import Dense
from convolutional import Convolutional
from reshape import Reshape
from activations import Sigmoid, Softmax
from losses import categorical_cross_entropy, categorical_cross_entropy_prime
from network import train, predict

start_time = time.time()

x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

labels = [0, 5]

x_train, y_train = preprocess_data(x_train, y_train, selected_classes=labels, limit_per_class=60_000)
x_test, y_test = preprocess_data(x_test, y_test, selected_classes=labels, limit_per_class=10_000)

network = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    MaxPooling(pool_size=(2, 2)),
    Reshape((5, 13, 13), (5 * 13 * 13, 1)),
    Dense(5 * 13 * 13, 100),
    Sigmoid(),
    Dropout(rate=0.5),
    Dense(100, len(labels)),
    Softmax()
]

train(
    network,
    categorical_cross_entropy,
    categorical_cross_entropy_prime,
    x_train,
    y_train,
    x_test=x_test,
    y_test=y_test,
    epochs=10,
    learning_rate=0.1
)


def calculate_accuracy(network, x_test, y_test):
    correct_predictions = 0
    for x, y in zip(x_test, y_test):
        output = predict(network, x)
        if np.argmax(output) == np.argmax(y):
            correct_predictions += 1

    accuracy = correct_predictions / len(x_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy


calculate_accuracy(network, x_test, y_test)
