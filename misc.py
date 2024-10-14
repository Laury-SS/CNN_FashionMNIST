import numpy as np

from network import to_categorical_numpy


def preprocess_data(x, y, selected_classes, limit_per_class=None):
    selected_indices = []
    class_mapping = {original_class: new_class for new_class, original_class in enumerate(selected_classes)}

    for class_label in selected_classes:
        class_indices = np.where(y == class_label)[0]

        if limit_per_class:
            class_indices = class_indices[:limit_per_class]

        selected_indices.append(class_indices)

    selected_indices = np.hstack(selected_indices)

    np.random.shuffle(selected_indices)

    x, y = x[selected_indices], y[selected_indices]

    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255

    y = np.array([class_mapping[label] for label in y])

    y = to_categorical_numpy(y, num_classes=len(selected_classes))

    y = y.reshape(len(y), len(selected_classes), 1)

    return x, y


def preprocess_data1(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical_numpy(y)
    y = y.reshape(len(y), 2, 1)
    return x, y
