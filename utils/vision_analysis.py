import pickle
import tensorflow as tf
import editdistance
import numpy as np
import scipy.spatial.distance as distance
from utils.config import get_indices, get_config
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


def calculate_and_save_rsa_scores(path, senders, receivers, cnn_sender, cnn_receiver,
                                  agent_type='both', flexible_role=False, n_examples=50):
    """Scores are calculated between the visual representations and the input (all attributes and individual attributes)
    In addition, scores are calculated between the two agents before and after training.

    :param path: path for storing the results
    :param senders: list of sender agents
    :param receivers: list of receiver agents
    :param cnn_sender: original vision module of the senders (before training)
    :param cnn_receiver: original vision module of the receivers (before training)
    :param agent_type: which agent to calculate the scores for ('sender', 'receiver', or 'both')
    :param flexible_role: whether agents are flexible-role agents
    :param n_examples: number of examples per class to use for calculating the scores

    :return: None --> save the rsa scores in the respective results file for each run
    """

    print('calculating rsa scores')

    images, attributes = collect_examples_per_class(n_examples=n_examples)
    color_attributes = attributes[:, 0:4]
    scale_attributes = attributes[:, 4:8]
    shape_attributes = attributes[:, 8:12]

    if flexible_role:
        assert len(senders) == 1 and len(receivers) == 1, "not implemented for flexible role + population"
        name1 = 'agent1'
        name2 = 'agent2'
    else:
        name1 = 'sender'
        name2 = 'receiver'

    correlation_scores = dict()
    features_sender_orig = cnn_sender(images).numpy()
    features_receiver_orig = cnn_receiver(images).numpy()

    all_features_sender_trained = []
    all_features_receiver_trained = []

    for i in range(len(senders)):
        print(name1)

        if len(receivers) > 1:
            S_append = str(i)
        else:
            S_append = ''

        if senders[0].vision_module:
            features_sender_trained = senders[0].vision_module(images).numpy()
        else:  # if only the vision module of the receiver is trained (language learning scenario)
            features_sender_trained = features_sender_orig
        all_features_sender_trained.append(features_sender_trained)

        if agent_type == 'both' or agent_type == 'sender':
            correlation_scores['rsa_' + name1 + S_append + '_attributes'] = cps.compute_similarity(
                attributes, features_sender_trained, distance.cosine, distance.cosine
            )
            correlation_scores['rsa_' + name1 + S_append + '_color'] = cps.compute_similarity(
                color_attributes, features_sender_trained, distance.cosine, distance.cosine
            )
            correlation_scores['rsa_' + name1 + S_append + '_scale'] = cps.compute_similarity(
                scale_attributes, features_sender_trained, distance.cosine, distance.cosine
            )
            correlation_scores['rsa_' + name1 + S_append + '_shape'] = cps.compute_similarity(
                shape_attributes, features_sender_trained, distance.cosine, distance.cosine
            )

    for i in range(len(receivers)):
        print(name2)

        if len(receivers) > 1:
            R_append = str(i)
        else:
            R_append = ''

        if receivers[i].vision_module:
            features_receiver_trained = receivers[i].vision_module(images).numpy()
        else:  # if only the vision module of the sender is trained (never happens in our simulations)
            features_receiver_trained = features_receiver_orig
        all_features_receiver_trained.append(features_receiver_trained)

        if agent_type == 'both' or agent_type == 'receiver':
            correlation_scores['rsa_' + name2 + R_append + '_attributes'] = cps.compute_similarity(
                attributes, features_receiver_trained, distance.cosine, distance.cosine
            )
            correlation_scores['rsa_' + name2 + R_append + '_color'] = cps.compute_similarity(
                color_attributes, features_receiver_trained, distance.cosine, distance.cosine
            )
            correlation_scores['rsa_' + name2 + R_append + '_scale'] = cps.compute_similarity(
                scale_attributes, features_receiver_trained, distance.cosine, distance.cosine
            )
            correlation_scores['rsa_' + name2 + R_append + '_shape'] = cps.compute_similarity(
                shape_attributes, features_receiver_trained, distance.cosine, distance.cosine
            )

    rsa_sender_receiver_orig = cps.compute_similarity(
        features_sender_orig, features_receiver_orig, distance.cosine, distance.cosine)
    correlation_scores['rsa_' + name1 + '_' + name2 + '_orig'] = rsa_sender_receiver_orig

    for i in range(len(senders)):
        S_append = str(i) if len(senders) > 1 else ''
    for j in range(len(receivers)):
        R_append = str(j) if len(receivers) > 1 else ''
        correlation_scores['rsa_' + name1 + S_append + '_' + name2 + R_append + '_trained'] = cps.compute_similarity(
            all_features_sender_trained[i], all_features_receiver_trained[j], distance.cosine, distance.cosine)

    print(correlation_scores)
    pickle.dump(correlation_scores, open(path + 'correlated_similarity_vision.pkl', 'wb'))
