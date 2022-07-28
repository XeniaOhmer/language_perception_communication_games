import numpy as np
from utils.config import get_attribute_dict
from sklearn.metrics import mutual_info_score as miscore
from scipy.stats import entropy


def get_padded_str_from_int_list(label_list):
    """ convert list of inputs to list of padded string.

    :param label_list: list of inputs
    :return: list of padded string
    """

    labels_str = [str(label) for label in label_list]
    max_len_label = max([len(label) for label in labels_str])
    labels_str_padded = [label.ljust(max_len_label, 'X') for label in labels_str]

    return labels_str_padded


def calc_entropy(X, base=None):
    """calculate entropy for data in X

    :param X: array of data / labels
    :param base: base used for logarithm
    :return: entropy
    """

    value, counts = np.unique(X, return_counts=True)  # get distribution of values

    return entropy(counts, base=base)


def joint_entropy(X, Y, base=None):
    """ calculate joint entropy between X and Y
    :param X: array of data / labels
    :param Y: array of data /labels
    :param base: base used for logarithm
    :return: joint entropy
    """

    XY = np.array([str(X[i]) + str(Y[i]) for i in range(len(X))])
    value, counts = np.unique(XY, return_counts=True)

    return entropy(counts, base=base)


def conditional_entropy(X, Y, base=None, normalizer='marginal'):
    """ calculate conditional entropy of X given Y

    :param X: array of data / labels
    :param Y: array of data / labels
    :param base: base used for logarithm
    :param normalizer: normalization method for conditional entropy (string)
                        marginal - divide by marginal entropy of X
                        arithmetic - divide by arithmetic mean of entropy X and entropy Y
                        joint - divide by the joint entropy between X and Y
    :return: (normalized) conditional entropy
    """

    X_given_Y = joint_entropy(X, Y, base=base) - calc_entropy(Y, base=base)
    if normalizer is None:
        normalized_entropy = X_given_Y
    elif normalizer == 'marginal':
        normalized_entropy = X_given_Y / calc_entropy(X, base=base)
    if normalizer == 'arithmetic':
        normalized_entropy = X_given_Y / (0.5 * (calc_entropy(Y, base=base) + calc_entropy(X, base=base)))
    elif normalizer == 'joint':
        normalized_entropy = X_given_Y / joint_entropy(X, Y, base=base)

    return normalized_entropy


def conditional_mi(X, Y, Z, normalizer='marginal'):
    """ conditional mutual information of X and Y given Z

    :param X: array of data / labels
    :param Y: array of data / labels
    :param Z: array of data / labels
    :param normalizer: normalization method for the conditional entropy (string)
    :return: (normalized) conditional mutual information
    """

    H_X_given_Z = conditional_entropy(X, Y, normalizer=None)
    H_Y_given_Z = conditional_entropy(Y, Z, normalizer=None)
    XY = np.array([str(X[i]) + str(Y[i]) for i in range(len(X))])
    H_XY_given_Z = conditional_entropy(XY, Z, normalizer=None)
    cond_mi = H_X_given_Z + H_Y_given_Z - H_XY_given_Z
    if normalizer is None:
        cond_mi = cond_mi
    elif normalizer == 'marginal':
        cond_mi = cond_mi / miscore(X, Y)

    return cond_mi


def mutual_info_threeRV(X, Y, Z, normalizer='arithmetic'):
    """ mutual information between three random variables

    :param X: array of data / labels
    :param Y: array of data / labels
    :param Z: array of data / labels
    :param normalizer: normalization method for the three-way mutual information (string)
                        only 'arithmetic possible' --> arithmetic mean of the entropies of X, Y, and Z
    :return: (normalized) mutual information
    """

    mi_XY = miscore(X, Y)
    cond_mi = conditional_mi(X, Y, Z)
    mi_XYZ = mi_XY - cond_mi
    if normalizer is None:
        mi_XYZ = mi_XYZ
    elif normalizer == 'arithmetic':
        mi_XYZ = mi_XYZ / ((1 / 3) * calc_entropy(X) + calc_entropy(Y) + calc_entropy(Z))

    return mi_XYZ


def conditional_entropy_threeRV(X, Y, Z, normalizer=None):
    """ conditional entropy of X given Y and Z

    :param X: array of data / labels
    :param Y: array of data / labels
    :param Z: array of data / labels
    :param normalizer: normalization method for the three-way conditional entropy (string)
                        only 'marginal' possible --> marginal entropy of X
    :return:
    """

    H_XYZ = calc_entropy([str(X[i]) + str(Y[i]) + str(Z[i]) for i in range(len(X))])
    H_Y_given_Z = conditional_entropy(Y, Z, normalizer=normalizer)
    H_Z = calc_entropy(Z)
    H_X_given_YZ = H_XYZ - H_Y_given_Z - H_Z
    if normalizer == 'marginal':
        H_X_given_YZ = H_X_given_YZ / calc_entropy(X)
    elif normalizer is None:
        H_X_given_YZ = H_X_given_YZ

    return H_X_given_YZ


def conditional_metric(X, Y, normalizer='marginal'):
    """ calculates effectiveness and efficiency scores by calculating 1 - normalized conditional entropy

    :param X: array of data / labels
    :param Y: array of data / labels
    :param normalizer: normalizer for conditional entropy function (string)
    :return: effectiveness or efficiency score (depending on the inputs)
    """

    return 1 - conditional_entropy(X, Y, normalizer=normalizer)


def get_message_labels(messages):
    """ label different messages, such that the same messages have the same labels
    @param messages: input messages (nxd array)
    @return: message label list
    """
    message_labels = []
    message_label_dict = {}
    messages = np.array(messages[:])
    label = -1
    for m in messages:
        if str(m) in message_label_dict.keys():
            message_labels.append(message_label_dict[str(m)])
        else:
            label = label + 1
            message_labels.append(label)
            message_label_dict[str(m)] = label
    return message_labels


class EntropyScores:
    """ Calculate entropy based scores: effectiveness, efficiency, positional disentanglement,
    bag of symbol disentanglement and residual entropy."""

    def __init__(self, messages, labels):
        """
        :param messages: messages generated by the sender
        :param labels: labels of the input objects
        """

        super().__init__()
        attribute_dict = get_attribute_dict()
        labels = np.argmax(labels, axis=1)

        self.attributes = np.array([attribute_dict[label] for label in labels])
        self.messages = messages
        self.message_length = messages.shape[1]
        message_labels = get_message_labels(messages)
        self.message_labels = get_padded_str_from_int_list(message_labels)
        self.input_labels = get_padded_str_from_int_list(labels)
        self.color_labels = get_padded_str_from_int_list(np.argmax(self.attributes[:, 0:4], axis=1) % 4)
        self.scale_labels = get_padded_str_from_int_list(np.argmax(self.attributes[:, 4:8], axis=1) % 4)
        self.shape_labels = get_padded_str_from_int_list(np.argmax(self.attributes[:, 8:12], axis=1) % 4)

    def calc_effectiveness(self):
        """calculate effectiveness scores for symbolic encodings of all attributes and individual attributes"""
        effectiveness = dict()
        effectiveness['color'] = conditional_metric(self.color_labels, self.message_labels, normalizer='marginal')
        effectiveness['shape'] = conditional_metric(self.shape_labels, self.message_labels, normalizer='marginal')
        effectiveness['scale'] = conditional_metric(self.scale_labels, self.message_labels, normalizer='marginal')
        effectiveness['all'] = conditional_metric(self.input_labels, self.message_labels, normalizer='marginal')
        return effectiveness

    def calc_efficiency(self):
        """calculate efficiency scores for symbolic encodings of all attributes and individual attributes"""
        efficiency = dict()
        efficiency['color'] = conditional_metric(self.message_labels, self.color_labels, normalizer='marginal')
        efficiency['shape'] = conditional_metric(self.message_labels, self.shape_labels, normalizer='marginal')
        efficiency['scale'] = conditional_metric(self.message_labels, self.scale_labels, normalizer='marginal')
        efficiency['all'] = conditional_metric(self.message_labels, self.input_labels, normalizer='marginal')
        return efficiency

    def calc_all_scores(self):
        """calculate effectiveness and efficiency scores"""
        effectiveness = self.calc_effectiveness()
        efficiency = self.calc_efficiency()
        all_scores = {
            'effectiveness': effectiveness,
            'efficiency': efficiency,
        }
        return all_scores


class ThreeWayEntropyScores():
    """ Calculate entropy based scores: effectiveness, efficiency, positional disentanglement,
    bag of symbol disentanglement and residual entropy."""

    def __init__(self, messages, input_labels, selection_labels):
        """
        :param messages: messages generated by the sender
        :param input_labels: labels of the input objects
        :param selection_labels: labels of the selected objects (non-hot)
        """

        super().__init__()
        attribute_dict = get_attribute_dict()
        input_labels = np.argmax(input_labels, axis=1)
        self.attributes = np.array([attribute_dict[label] for label in input_labels])
        self.messages = messages
        self.message_length = messages.shape[1]
        message_labels = get_message_labels(messages)
        self.message_labels = get_padded_str_from_int_list(message_labels)
        self.input_labels = get_padded_str_from_int_list(input_labels)
        self.selection_labels = get_padded_str_from_int_list(selection_labels)

    def calc_I_OS_given_M(self):
        """ calculate mutual information between objects and selections given messages """
        return conditional_mi(self.input_labels, self.selection_labels, self.message_labels)

    def calc_I_OM(self):
        """ calculate mutual information between objects and messages """
        return miscore(self.input_labels, self.message_labels)

    def calc_I_MS(self):
        """ calculate mutual information between messages and selections """
        return miscore(self.message_labels, self.selection_labels)

    def calc_all_scores(self):
        """calculate effectiveness and efficiency scores"""
        I_OM = self.calc_I_OM()
        I_MS = self.calc_I_MS()
        I_OS_given_M = self.calc_I_OS_given_M()
        all_scores = {
            'I_OM': I_OM,
            'I_MS': I_MS,
            'I_OS_given_M': I_OS_given_M
        }
        return all_scores
