import numpy as np
import matplotlib.pyplot as plt


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def train(network, loss_function, loss_prime, x_train, y_train, x_test=None, y_test=None, epochs=10, learning_rate=0.1):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        train_loss = 0
        correct_train_predictions = 0
        for x, y in zip(x_train, y_train):
            output = predict(network, x)
            train_loss += loss_function(y, output)

            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

            if np.argmax(output) == np.argmax(y):
                correct_train_predictions += 1

        train_loss /= len(x_train)
        train_accuracy = correct_train_predictions / len(x_train)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        if x_test is not None and y_test is not None:
            test_loss, test_accuracy = evaluate(network, loss_function, x_test, y_test)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy * 100:.2f}%")
        with open("resultado.txt", "a") as resultado:
            resultado.write(f"{epoch + 1};{train_loss};{train_accuracy};")

        if x_test is not None and y_test is not None:
            print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy * 100:.2f}%")

            with open("resultado.txt", "a") as resultado:
                resultado.write(f"{test_loss};{test_accuracy}\n")

    plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies)


def evaluate(network, loss_function, x_data, y_data):
    total_loss = 0
    correct_predictions = 0
    for x, y in zip(x_data, y_data):
        output = predict(network, x)
        total_loss += loss_function(y, output)

        if np.argmax(output) == np.argmax(y):
            correct_predictions += 1

    average_loss = total_loss / len(x_data)
    accuracy = correct_predictions / len(x_data)
    return average_loss, accuracy


def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def to_categorical_numpy(y, num_classes=None):
    """
    Converts a class vector (integers) to binary class matrix (one-hot encoding).

    Parameters:
    y: class vector to be converted into a matrix (integers).
    num_classes: total number of classes (if None, it is inferred from the data).

    Returns:
    A binary matrix representation of the input class vector.
    """
    y = np.array(y, dtype='int')  # Ensure the input is an array
    if not num_classes:
        num_classes = np.max(y) + 1  # Infers the number of classes if not provided
    categorical = np.zeros((y.shape[0], num_classes))  # Create a zeros matrix
    categorical[np.arange(y.shape[0]), y] = 1  # Set the appropriate index to 1
    return categorical
