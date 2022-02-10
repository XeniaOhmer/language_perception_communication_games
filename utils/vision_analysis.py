import pickle
import tensorflow as tf
import editdistance
import numpy as np
import scipy.spatial.distance as distance
from utils.config import get_indices, get_config
from nn.agents import Sender, Receiver
from utils.rsa import CorrelatedPairwiseSimilarity as cps
from utils.load_data import collect_examples_per_class

# collect data set meta-information
all_cnn_paths, image_dim, n_classes, feature_dims = get_config()
color_indices, scale_indices, shape_indices = get_indices()

# for each color, shape, and scale, collect classes of different color, shape and scale respectively
all_indices = np.arange(64)
other_color_indices = [all_indices[~np.isin(all_indices, color_indices[i])] for i in range(4)]
other_shape_indices = [all_indices[~np.isin(all_indices, shape_indices[i])] for i in range(4)]
other_scale_indices = [all_indices[~np.isin(all_indices, scale_indices[i])] for i in range(4)]


def featurewise_similarity(similarity_matrix):
    """ calculate similarities to other classes sharing the same feature value for either color, size or shape
    :param similarity_matrix: similarity matrix as calculated with class_similarity_matrix
    :return: for each class, similarities with respect to other classes sharing the same feature for either color
             size or shape --> dimensionality is [n_classes, n_features], here [64,3]
    """

    similarities = np.zeros((n_classes, 3))

    for c in range(n_classes):

        for color_ind in color_indices:
            color_ind_list = list(color_ind)
            if c in color_ind:
                color_ind_list.remove(c)
                similarities[c, 0] = np.mean(similarity_matrix[c, color_ind_list])

        for scale_ind in scale_indices:
            scale_ind_list = list(scale_ind)
            if c in scale_ind:
                scale_ind_list.remove(c)
                similarities[c, 1] = np.mean(similarity_matrix[c, scale_ind_list])

        for shape_ind in shape_indices:
            shape_ind_list = list(shape_ind)
            if c in shape_ind:
                shape_ind_list.remove(c)
                similarities[c, 2] = np.mean(similarity_matrix[c, shape_ind_list])

    return similarities


def featurewise_dissimilarity(similarity_matrix):
    """ calculate similarities to other classes NOT sharing the same feature value for either color, size or shape
        :param similarity_matrix: similarity matrix as calculated with class_similarity_matrix
        :return:    for each class, similarities with respect to other classes NOT sharing the same feature for either
                    color size or shape --> dimensionality is [n_classes, n_features], here [64,3]
                    
        commented part was used to correct the all bias values for mutual attenutation
    """

    similarities = np.zeros((n_classes, 3))

    for c in range(n_classes):

        for j, color_ind in enumerate(color_indices):
            color_ind_list = list(other_color_indices[j])
            if c in color_ind:
                similarities[c, 0] = np.mean(similarity_matrix[c, color_ind_list])

        for j, scale_ind in enumerate(scale_indices):
            scale_ind_list = list(other_scale_indices[j])
            if c in scale_ind:
                similarities[c, 1] = np.mean(similarity_matrix[c, scale_ind_list])

        for j, shape_ind in enumerate(shape_indices):
            shape_ind_list = list(other_shape_indices[j])
            if c in shape_ind:
                similarities[c, 2] = np.mean(similarity_matrix[c, shape_ind_list])

    return similarities


def class_similarity_matrix(features, n_examples, metric='cosine', message_length=3):
    """ calculates matrix with pairwise object (class) similarities
    :param features:    CNN features extracted for objects, sorted by class (first n_examples belong to class 0 etc)
    :param n_examples:  number of examples per class
    :param metric: which distance function to use
    :param message_length: length of the messages
    :return:            matrix with pairwise class similarities, entry i,j is similarity between class i and class j
    """

    if metric == 'cosine':
        similarity = 1 - distance.pdist(features, 'cosine')
    elif metric == 'edit':
        distance_fn = lambda x, y: editdistance.eval(x, y)
        similarity = 1 - distance.pdist(features, distance_fn) / message_length

    similarity = distance.squareform(similarity)

    np.fill_diagonal(similarity, np.nan)

    sim_matrix = np.zeros((n_classes, n_classes))
    for ind_i, i in enumerate(range(0, n_classes * n_examples, n_examples)):
        for ind_j, j in enumerate(range(0, n_classes * n_examples, n_examples)):
            sim_matrix[ind_i, ind_j] = np.nanmean(similarity[i:i + n_examples, j:j + n_examples])
    return sim_matrix


def get_vision_module_similarity_matrices(cnn_keys, n_examples=50, layer=0):
    """ calculate vision module biases for each feature, and plot the similarity matrices.

    :param cnn_keys: keys for calling the CNNs
    :param n_examples: numbers of examples used to calculate the scores
    :param layer: which dense layer of the network
    :return: list of similarity matrices
    """

    images, attributes = collect_examples_per_class(n_examples=n_examples)

    sim_matrices = []
    for cnn_key in cnn_keys:
        path_vision = all_cnn_paths[cnn_key]
        vision = tf.keras.models.load_model(path_vision)
        vision = tf.keras.Model(inputs=vision.input, outputs=vision.get_layer(['dense_1', 'dense_2'][layer]).output)

        features = np.reshape(vision(images), (len(images), -1))
        similarity_matrix = class_similarity_matrix(features, n_examples)
        sim_matrices.append(similarity_matrix)

    return sim_matrices


def save_rsa_cnns(n_examples=50, mode='basic', sf=None, tw=None):
    """calculate and store RSA scores of pretrained CNNs across attributes and for each attributes.

    :param n_examples: number of examples per class used for calculating the scores
    :param mode:    'basic' --> evaluate all 5 basic conditions: default, color, scale, shape, all
                    'mixed' --> evaluate all mixed models: color-scale, color-shape, scale-shape
    :param sf: smoothing factor, only relevant for mixed models
    :param tw: weight between the two biases, only relevant for mixed models
    :return: nothing, but store rsa scores for all attributes and individual attributes
    """

    images, attributes = collect_examples_per_class(n_examples=n_examples)

    if mode == 'basic':
        name = 'basic_'
        cnns = ['default0-0', 'color0-6', 'scale0-6', 'shape0-6', 'all0-8']
    elif mode == 'mixed':
        name = 'mixed_tw-' + tw + '_nonlinear_sf-' + str(sf) + '_'
        cnns = ['color-size', 'color-shape', 'shape-size']

    rsa = {}
    for cnn in cnns:

        if mode == 'basic':
            path_vision = all_cnn_paths[cnn]
        elif mode == 'mixed':
            path_vision = ('share/mixed/finalweights_lt-coarse_trait-' +
                           cnn + '_tw-' + tw + '_sf-' + sf + '.hdf5')
        vision = tf.keras.models.load_model(path_vision)
        vision_module = tf.keras.Model(inputs=vision.input, outputs=vision.get_layer('dense_1').output)

        rsa[cnn] = {}

        rsa_all = cps.compute_similarity(vision_module(images), attributes, distance.cosine, distance.cosine)
        rsa_color = cps.compute_similarity(vision_module(images), attributes[:, 0:4], distance.cosine, distance.cosine)
        rsa_scale = cps.compute_similarity(vision_module(images), attributes[:, 4:8], distance.cosine, distance.cosine)
        rsa_shape = cps.compute_similarity(vision_module(images), attributes[:, 8:12], distance.cosine, distance.cosine)

        rsa[cnn]['all'] = rsa_all
        rsa[cnn]['color'] = rsa_color
        rsa[cnn]['scale'] = rsa_scale
        rsa[cnn]['shape'] = rsa_shape

    file = 'results/rsa_attributes_CNNfeatures/' + name + str(n_examples) + 'examples.pkl'
    pickle.dump(rsa, open(file, 'wb'))


def calculate_and_save_rsa_scores(mode, conditions, cnn_keys_sender, cnn_keys_receiver, vs=4, n_runs=10, n_epochs=150,
                                  agent='both', n_examples=50):
    """Calculate and save rsa scores for several runs.
    Scores are calculated between the visual representations and the input (all attributes, and individual attributes.
    In addition, scores are calculated between the two agents before and after training.

    :param mode: training setting (e.g. 'language_emergence_basic')
    :param conditions: bias conditions (e.g. ['default', 'color', ...]
    :param cnn_keys_sender: keys of the sender CNNs
    :param cnn_keys_receiver: keys of the receiver CNNs
    :param vs: vocabulary size
    :param n_runs: number of runs per condition
    :param n_epochs: number of training epochs
    :param agent: for which agent to calculate the scores ('sender', 'receiver', or 'both')
    :param n_examples: number of examples per class to use for calculating the scores
    :return: None --> save the rsa scores in the respective results file for each run
    """
    images, attributes = collect_examples_per_class(n_examples=n_examples)
    color_attributes = attributes[:, 0:4]
    scale_attributes = attributes[:, 4:8]
    shape_attributes = attributes[:, 8:12]

    for c, condition in enumerate(conditions):

        for run in range(n_runs):

            path = 'results/' + mode + '/' + condition + str(run) + '/vs' + str(vs) + '_ml3/'

            correlation_scores = dict()

            cnn_sender = tf.keras.models.load_model(all_cnn_paths[cnn_keys_sender[c]])
            cnn_sender = tf.keras.Model(inputs=cnn_sender.input,
                                        outputs=cnn_sender.get_layer('dense_1').output)
            features_sender_orig = cnn_sender(images).numpy()

            sender = Sender(vs, 3, 128, 128, cnn_sender)
            sender.load_weights(path + 'sender_weights_epoch' + str(n_epochs - 1) + '/')
            features_sender_trained = sender.vision_module(images).numpy()

            cnn_receiver = tf.keras.models.load_model(all_cnn_paths[cnn_keys_receiver[c]])
            cnn_receiver = tf.keras.Model(inputs=cnn_receiver.input,
                                          outputs=cnn_receiver.get_layer('dense_1').output)
            features_receiver_orig = cnn_receiver(images).numpy()

            receiver = Receiver(vs, 3, 128, 128, cnn_receiver, n_distractors=2)
            receiver.load_weights(path + 'receiver_weights_epoch' + str(n_epochs - 1) + '/')
            features_receiver_trained = receiver.vision_module(images).numpy()

            if agent == 'both' or agent == 'sender':
                correlation_scores['rsa_sender_attributes'] = cps.compute_similarity(
                    attributes, features_sender_trained, distance.cosine, distance.cosine
                )
                correlation_scores['rsa_sender_color'] = cps.compute_similarity(
                    color_attributes, features_sender_trained, distance.cosine, distance.cosine
                )
                correlation_scores['rsa_sender_scale'] = cps.compute_similarity(
                    scale_attributes, features_sender_trained, distance.cosine, distance.cosine
                )
                correlation_scores['rsa_sender_shape'] = cps.compute_similarity(
                    shape_attributes, features_sender_trained, distance.cosine, distance.cosine
                )

            if agent == 'both' or agent == 'receiver':
                correlation_scores['rsa_receiver_attributes'] = cps.compute_similarity(
                    attributes, features_receiver_trained, distance.cosine, distance.cosine
                )
                correlation_scores['rsa_receiver_color'] = cps.compute_similarity(
                    color_attributes, features_receiver_trained, distance.cosine, distance.cosine
                )
                correlation_scores['rsa_receiver_scale'] = cps.compute_similarity(
                    scale_attributes, features_receiver_trained, distance.cosine, distance.cosine
                )
                correlation_scores['rsa_receiver_shape'] = cps.compute_similarity(
                    shape_attributes, features_receiver_trained, distance.cosine, distance.cosine
                )

            # only needs to be calculated once since its the same for all runs
            if run == 0:
                rsa_sender_receiver_orig = cps.compute_similarity(
                    features_sender_orig, features_receiver_orig, distance.cosine, distance.cosine)
                correlation_scores['rsa_sender_receiver_orig'] = rsa_sender_receiver_orig
            else:
                correlation_scores['rsa_sender_receiver_orig'] = rsa_sender_receiver_orig

            correlation_scores['rsa_sender_receiver_trained'] = cps.compute_similarity(
                features_sender_trained, features_receiver_trained, distance.cosine, distance.cosine)

            pickle.dump(correlation_scores, open(path + 'correlated_similarity_vision.pkl', 'wb'))
