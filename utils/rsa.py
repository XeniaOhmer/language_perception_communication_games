import numpy as np
from scipy.spatial import distance
from scipy.stats import spearmanr
from utils.config import get_attribute_dict


def labels_to_attributes(labels):
    """ translates between object labels (so classes) and the corresponding class-defining attributes
    :param labels: object label (non-hot)
    :param dataset: dataset
    :return:    returns a list containing the attributes for each object class as a k-hot vector
                see communication_game.config.py
    """

    attribute_dict = get_attribute_dict()

    return [attribute_dict[l] for l in labels]


class CorrelatedPairwiseSimilarity:
    """ Base class for calculating topographic similarity and RSA."""

    def __init__(self):
        pass

    @staticmethod
    def compute_similarity(input1, input2, distance_fn1, distance_fn2):
        """ generates the similarity / distance matrices for two data sets, and then correlates these matrices

        :param input1: data set 1 (n x d array)
        :param input2: data set 2 (n x d array)
        :param distance_fn1: distance function to be used on data set 1
        :param distance_fn2: distance function to be used on data set 2
        :return: the correlation score for the two distance matrices
        """

        dist1 = distance.pdist(input1, distance_fn1)
        dist2 = distance.pdist(input2, distance_fn2)

        nan_prop1 = np.count_nonzero(np.isnan(dist1)) / len(dist1)
        nan_prop2 = np.count_nonzero(np.isnan(dist2)) / len(dist2)
        if nan_prop1 > 0.05 or nan_prop2 > 0.05:
            rsa = None
        else:
            rsa = spearmanr(dist1, dist2, nan_policy='omit').correlation

        return rsa


class RSA(CorrelatedPairwiseSimilarity):
    """ Calculates the representational similarity analysis score for all pairwise combinations of sender space,
    receiver space, and input space. Calculation is essentially the same as for topographic similarity.
    """

    def __init__(self, sender, receiver, dist=distance.cosine):
        super().__init__()
        self.distance = dist
        self.sender = sender
        self.receiver = receiver

    def get_all_RSAs(self, attributes, sender_input):
        """ calculate the RSA score between every combination of sender hidden state, receiver hidden state,
        and the symbolic input representations.

        :param attributes: k-hot attribute encodings of the input
        :param sender_input: input images
        :return:
        """
        messages, _, _, _, hidden_sender = self.sender.forward(sender_input, training=False)
        RSA_sender_input = self.compute_similarity(attributes, hidden_sender, self.distance, self.distance)
        hidden_receiver = self.receiver.language_module(messages)
        RSA_receiver_input = self.compute_similarity(attributes, hidden_receiver, self.distance, self.distance)
        RSA_sender_receiver = self.compute_similarity(hidden_sender, hidden_receiver, self.distance, self.distance)
        return RSA_sender_input, RSA_receiver_input, RSA_sender_receiver

    def get_all_RSAs_precalc(self, attributes, hidden_sender, messages):
        """ Same as above, but here, the messages are already given which speeds up the computation.
        """
        RSA_sender_input = self.compute_similarity(attributes, hidden_sender, self.distance, self.distance)
        hidden_receiver = self.receiver.language_module(messages)
        RSA_receiver_input = self.compute_similarity(attributes, hidden_receiver, self.distance, self.distance)
        RSA_sender_receiver = self.compute_similarity(hidden_sender, hidden_receiver, self.distance, self.distance)
        return RSA_sender_input, RSA_receiver_input, RSA_sender_receiver