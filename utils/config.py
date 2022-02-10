import numpy as np

# Here, we collect specs of our pre-trained cnns and the data set

# paths to the pre-trained CNNs
cnn_paths = {

    'default0-0': 'share/all-weights-sfactor-0.00-200-0.0021.hdf5',
    'all0-6': 'share/all-weights-sfactor-0.60-200-1.0843.hdf5',
    'all0-8': 'share/all-weights-sfactor-0.80-200-1.7984.hdf5',
    'color0-6': 'share/objectHue-weights-sfactor-0.60-200-1.2947.hdf5',
    'scale0-6': 'share/scale-weights-sfactor-0.60-200-1.0954.hdf5',
    'shape0-6': 'share/shape-weights-sfactor-0.60-200-1.1110.hdf5',

    # mixed models, balanced biases

    'color-shape0-8': 'share/mixed/finalweights_lt-coarse_trait-color-shape_tw-35_sf-0.8.hdf5',
    'color-scale0-8': 'share/mixed/finalweights_lt-coarse_trait-color-size_tw-30_sf-0.8.hdf5',
    'scale-shape0-8': 'share/mixed/finalweights_lt-coarse_trait-shape-size_tw-25_sf-0.8.hdf5',

}

# different specs of our data set

# k-hot attribute value encoding for each object, the first four values indicate color,
# the second four values indicate scale, the last four values indicate shape
attribute_dict = {

    0: [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    1: [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    2: [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    3: [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    4: [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    5: [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    6: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    7: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    8: [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    9: [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    10: [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    11: [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    12: [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    13: [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    14: [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    15: [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    16: [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    17: [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    18: [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    19: [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    20: [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    21: [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    22: [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    23: [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    24: [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    25: [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    26: [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    27: [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    28: [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    29: [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    30: [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    31: [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    32: [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    33: [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    34: [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    35: [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    36: [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    37: [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    38: [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    39: [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    40: [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    41: [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    42: [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    43: [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    44: [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    45: [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    46: [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    47: [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    48: [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
    49: [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
    50: [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
    51: [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    52: [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
    53: [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
    54: [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
    55: [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    56: [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
    57: [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    58: [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    59: [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
    60: [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
    61: [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    62: [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    63: [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]
}

image_dim = 64
n_classes = 64
feature_dims = [4, 4, 4]
feature_order = ('color', 'scale', 'shape')

# class indices for each color, scale, and shape value
# each list contains 4 arrays, each array gives the objects (class indices) that have the same value for this attribute
# color_indices: [object indices with color value 0, object indices with color value 1, ...]
color_indices = [np.arange(0, 16), np.arange(16, 32), np.arange(32, 48), np.arange(48, 64)]

scale_indices = [np.concatenate([[0 + x, 1 + x, 2 + x, 3 + x] for x in range(0, 64, 16)]),
                 np.concatenate([[0 + x, 1 + x, 2 + x, 3 + x] for x in range(4, 64, 16)]),
                 np.concatenate([[0 + x, 1 + x, 2 + x, 3 + x] for x in range(8, 64, 16)]),
                 np.concatenate([[0 + x, 1 + x, 2 + x, 3 + x] for x in range(12, 64, 16)])]

shape_indices = [np.arange(0, 64, 4), np.arange(1, 64, 4), np.arange(2, 64, 4), np.arange(3, 64, 4)]


def get_feature_information():
    return feature_dims, feature_order


def get_attribute_dict():
    return attribute_dict


def get_cnn_paths():
    return cnn_paths


def get_indices():
    return color_indices, scale_indices, shape_indices


def get_config():
    return cnn_paths, image_dim, n_classes, feature_dims


def get_smoothed_labels(train_labels, sim_sender, sim_receiver, sf_sender, sf_receiver):
    """ function to get smoothed labels for different biases and smoothing factors

    :param train_labels: labels of the training data, one hot encoded
    :param sim_sender: vision bias of sender ('default', 'color', 'scale', 'shape', 'all)
    :param sim_receiver: vision bias of receiver ('default', 'color', 'scale', 'shape', 'all)
    :param sf_sender: smoothing factor sender
    :param sf_receiver: smoothing factor receiver
    :return: smoothed labels for sender and receiver
    """

    sf_s = round(np.int(sf_sender.split('-')[1]) * 0.1, 2)
    sf_r = round(np.int(sf_receiver.split('-')[1]) * 0.1, 2)

    labels_nonhot = np.argmax(train_labels, axis=1)

    train_labels_new = []
    for agent, sim_agent in enumerate([sim_sender, sim_receiver]):
        sf = [sf_s, sf_r][agent]

        if sim_agent == 'default':
            train_labels_agent = train_labels  # do nothing
        else:
            train_labels_agent = np.zeros_like(train_labels, dtype=np.float32)
            for i, label in enumerate(labels_nonhot):
                if sim_agent == 'color':
                    for colors in color_indices:
                        if label in colors:
                            train_labels_agent[i, colors] = sf / 15
                            train_labels_agent[i, label] = 1 - sf
                elif sim_agent == 'shape':
                    for shapes in shape_indices:
                        if label in shapes:
                            train_labels_agent[i, shapes] = sf / 15
                            train_labels_agent[i, label] = 1 - sf
                elif sim_agent == 'scale':
                    for scales in scale_indices:
                        if label in scales:
                            train_labels_agent[i, scales] = sf / 15
                            train_labels_agent[i, label] = 1 - sf
                elif sim_agent == 'all':
                    for index_list in [color_indices, scale_indices, shape_indices]:
                        for indices in index_list:
                            if label in indices:
                                train_labels_agent[i, indices] += sf / 45
                    train_labels_agent[i, label] = 1 - sf

        train_labels_new.append(train_labels_agent)

    return train_labels_new[0], train_labels_new[1]
