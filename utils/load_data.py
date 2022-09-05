import numpy as np
from utils.config import get_attribute_dict
from utils.train import load_data


def load_train():
    """ load the training data

    :return: data and labels (one hot encoded)
    """

    (data, labels), _, _, _ = load_data((64, 64, 3))

    return data, labels


def collect_examples_per_class(n_examples=50):
    """ returns a certain number of example images for each object class

    :param n_examples: number of example images
    :return: images and their symbolic (k-hot) attribute encodings
    """

    data, labels = load_train()
    labels = np.argmax(labels, axis=1)
    attribute_dict = get_attribute_dict()
    n_classes = 64

    # select 50 examples of each class
    images = np.zeros((n_classes * n_examples, 64, 64, 3), dtype=np.float32)

    for c in range(n_classes):
        images[c * n_examples:(c + 1) * n_examples] = data[labels == c][0:n_examples]

    attributes = np.reshape(np.array([[attribute_dict[i]] * n_examples for i in range(n_classes)]),
                            (n_examples * n_classes, 12))

    return images, attributes
