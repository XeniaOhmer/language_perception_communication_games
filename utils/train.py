import numpy as np
import pathlib
import h5py

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.compat.v1 import ConfigProto, Session

import argparse


def load_default_model_params(dataset='shapes'):
    model_params = dict()
    if dataset == 'shapes':
        model_params['conv_depths'] = [32, 32]
        model_params['fc_depths'] = [16, 16]
        model_params['conv_pool'] = [True, False]
    else:
        raise ValueError('Default model for dataset \'{}\' not defined.'.format(dataset))

    return model_params


def sum_label_components(labels, rel_comps=None, factor=0.1, verbose=False):
    if rel_comps is None:
        raise ValueError('must set kwarg rel_comps to use this function.')
    if verbose:
        orig_labels = labels.copy()

    labels = labels.astype('float')
    for (idx, l) in enumerate(labels):
        true_class = int(np.nonzero(l)[0])
        labels[idx] = (1 - factor) * l + factor * rel_comps[true_class]

    if verbose is True:
        print("[INFO] smoothing amount: {}".format(factor))
        print("[INFO] before smoothing: {}".format(orig_labels[0]))
        print("[INFO] after smoothing: {}".format(labels[0]))
        print("[INFO] compiling model...")

    return labels


def coarse_generic_relational_labels(label_map):
    num_classes = np.max([int(k) for k in label_map.keys()]) + 1
    output_dicts = [{} for _ in range(len(label_map[0]))]
    for i in range(num_classes):
        traits_i = label_map[i]
        tmp_labels = [np.zeros(num_classes) for _ in range(len(traits_i))]
        for j in range(num_classes):
            if i == j:
                continue
            traits_j = label_map[j]
            for ii in range(len(traits_j)):
                if traits_i[ii] == traits_j[ii]:
                    tmp_labels[ii][j] = 1

        for j in range(len(tmp_labels)):
            tmp_labels[j] /= np.sum(tmp_labels[j])
            output_dicts[j][i] = tmp_labels[j]

    output_dicts = tuple(output_dicts)

    return output_dicts


def linear_relational_labels(label_map, trait_names=None):
    num_classes = np.max([int(k) for k in label_map.keys()]) + 1
    output_dicts = [{} for _ in range(len(label_map[0]))]
    label_vals = [[] for _ in range(len(label_map[0]))]
    if trait_names is None:
        label_vals = [[] for _ in range(len(label_map[0]))]
    for k in label_map.keys():
        for i in range(len(label_vals)):
            label_vals[i].append(label_map[k][i])
    for i in range(len(label_vals)):
        label_vals[i] = sorted(set(label_vals[i]))
    max_sims = [0 for _ in range(len(label_vals))]
    for (l_idx, labels) in enumerate(label_vals):
        if trait_names[l_idx] == 'color':
            for (i, val1) in enumerate(labels):
                for val2 in labels[i + 1:]:
                    # calculates distance circularly for color
                    x = min(val1, val2)
                    y = max(val1, val2)
                    circular_diff = min(abs(x - y), abs(x - (y - 1)))
                    sim = 1 / abs(circular_diff)
                    if sim > max_sims[l_idx]:
                        max_sims[l_idx] = sim
        else:
            for (i, val1) in enumerate(labels):
                for val2 in labels[i + 1:]:
                    sim = 1 / abs(val1 - val2)
                    if sim > max_sims[l_idx]:
                        max_sims[l_idx] = sim
    equivalence_vals = [x * 2 for x in max_sims]

    for i in range(num_classes):
        traits_i = label_map[i]
        tmp_labels = [np.zeros(num_classes) for _ in range(len(traits_i))]
        for j in range(num_classes):
            if i == j:
                continue
            traits_j = label_map[j]
            for ii in range(len(traits_j)):
                if traits_i[ii] == traits_j[ii]:
                    tmp_labels[ii][j] = equivalence_vals[ii]
                else:
                    if trait_names[ii] == 'color':
                        # to do distance circularly since max value is adjacent to min value
                        x = min(traits_i[ii], traits_j[ii])
                        y = max(traits_i[ii], traits_j[ii])
                        circular_diff = min(abs(x - y), abs(x - (y - 1)))
                        tmp_labels[ii][j] = 1 / circular_diff
                    else:
                        tmp_labels[ii][j] = 1 / abs(traits_i[ii] - traits_j[ii])

        for j in range(len(tmp_labels)):
            tmp_labels[j] /= np.sum(tmp_labels[j])
            output_dicts[j][i] = tmp_labels[j]
    output_dicts = tuple(output_dicts)

    return output_dicts


def get_shape_color_labels(full_labels,
                           trait_idxs=(2, 3, 4),
                           balance_traits=True,
                           trait_weights=None,
                           balance_type=2,
                           label_type='coarse'):
    possible_values = [[] for _ in range(len(trait_idxs))]
    trait_names_by_idx = ['floorHue', 'wallHue', 'color', 'scale', 'shape', 'orientation']

    # default trait_idxs set to (2,3,4) corresponding to color, scale, and shape
    extracted_traits = [tuple(entry) for entry in list(full_labels[:, trait_idxs])]

    for tup in extracted_traits:
        for (idx, entry) in enumerate(tup):
            possible_values[idx].append(entry)
    for (idx, p) in enumerate(possible_values):
        possible_values[idx] = sorted(set(p))

    # since there were only 4 possible shapes, we extracted 4 approximately equally spaced
    # values from the other two traits. balance_type == 2 was used for our experiments. The first
    # list in idxes_to_keep selects values for color and the second list selects values for
    # the object scale, based on the configuration set by the extracted_traits variable
    if balance_traits:
        if balance_type == 0:
            idxes_to_keep = [[0, 3, 6, 9], [0, 3, 5, 7]]
        elif balance_type == 1:
            idxes_to_keep = [[1, 2, 4, 8], [0, 3, 5, 7]]
        elif balance_type == 2:
            idxes_to_keep = [[0, 2, 4, 8], [0, 3, 5, 7]]
        values_to_keep = [[], []]

        for idx in [0, 1]:
            for val_idx in idxes_to_keep[idx]:
                values_to_keep[idx].append(possible_values[idx][val_idx])
        filtered_traits = []
        keeper_idxs = []
        for (idx, traits) in enumerate(extracted_traits):
            if traits[0] in values_to_keep[0] and traits[1] in values_to_keep[1]:
                filtered_traits.append(traits)
                keeper_idxs.append(idx)
        extracted_traits = filtered_traits
    else:
        keeper_idxs = None

    trait_names = [trait_names_by_idx[i] for i in trait_idxs]
    unique_traits = sorted(set(extracted_traits))
    labels = np.zeros((len(extracted_traits), len(unique_traits)))

    # these dictionaries are used to convert between indices for one-hot target vectors
    # and the corresponding trait combination that that entry represents, which defines the class
    # composition of the classification problem
    label2trait_map = dict()
    trait2label_map = dict()

    for (i, traits) in enumerate(unique_traits):
        trait2label_map[traits] = i
        label2trait_map[i] = traits
    if label_type == 'coarse':
        labels_template = coarse_generic_relational_labels(label2trait_map)
    elif label_type == 'linear':
        labels_template = linear_relational_labels(label2trait_map,
                                                   trait_names=trait_names)
    test = coarse_generic_relational_labels(label2trait_map)
    relational_labels = dict()
    test_relational_labels = dict()
    # calculating for individual traits
    for (i, k) in enumerate(trait_names):
        relational_labels[k] = labels_template[i]
        test_relational_labels[k] = test[i]

    # calculating for dual traits
    if trait_weights is None:
        trait_weights = dict()
        trait_weights['color-shape'] = [0.5, 0.5]
        trait_weights['color-size'] = [0.5, 0.5]
        trait_weights['shape-size'] = [0.5, 0.5]
    elif type(trait_weights) is list:
        tw_list = trait_weights
        trait_weights = dict()
        for k in ['color-shape', 'color-size', 'shape-size']:
            trait_weights[k] = tw_list
        print('trait_weights = ')
        print(trait_weights)
    relational_labels['color-shape'] = dict()
    relational_labels['color-size'] = dict()
    relational_labels['shape-size'] = dict()

    for idx in labels_template[0].keys():
        relational_labels['color-shape'][idx] = 0
        relational_labels['color-size'][idx] = 0
        relational_labels['shape-size'][idx] = 0
        for (i, k) in enumerate(trait_names):
            if k == 'color':
                relational_labels['color-shape'][idx] += trait_weights['color-shape'][0] * labels_template[i][idx]
                relational_labels['color-size'][idx] += trait_weights['color-shape'][0] * labels_template[i][idx]
            elif k == 'scale':
                relational_labels['shape-size'][idx] += trait_weights['shape-size'][1] * labels_template[i][idx]
                relational_labels['color-size'][idx] += trait_weights['color-size'][1] * labels_template[i][idx]
            elif k == 'shape':
                relational_labels['shape-size'][idx] += trait_weights['shape-size'][0] * labels_template[i][idx]
                relational_labels['color-shape'][idx] += trait_weights['color-shape'][1] * labels_template[i][idx]

    # calculating for all traits
    relational_labels['all'] = dict()
    for k in labels_template[0].keys():
        relational_labels['all'][k] = 0
        for lab in labels_template:
            relational_labels['all'][k] += 1 / len(labels_template) * lab[k]

    # generating one-hot labels
    for (i, traits) in enumerate(extracted_traits):
        labels[i, trait2label_map[traits]] = 1
    return labels, relational_labels, keeper_idxs, trait_weights


def load_data(input_shape, normalize=True,
              subtract_mean=True,
              balance_traits=True,
              trait_weights=None,
              balance_type=2,
              return_trait_weights=False,
              return_full_labels=False,
              label_type='linear',
              datapath=None):
    assert return_trait_weights + return_full_labels < 2, 'only can return one of trait_weights or full_labels'

    if datapath is None:
        data_path = 'data/3dshapes.h5'
    else:
        data_path = datapath
    parent_dir = str(pathlib.Path().absolute()).split('/')[-1]
    if parent_dir == 'SimilarityGames':
        data_path = data_path[3:]
    dataset = h5py.File(data_path, 'r')
    data = dataset['images'][:]
    full_labels = dataset['labels'][:]
    labels_reg, labels_relational, keeper_idxs, trait_weights = get_shape_color_labels(full_labels,
                                                                                       balance_traits=balance_traits,
                                                                                       balance_type=balance_type,
                                                                                       label_type=label_type,
                                                                                       trait_weights=trait_weights)

    # chooses one of 3 variables to return as the meta variable - note, only one of the boolean return
    # variables should be set to True
    if return_full_labels:
        meta = full_labels
    elif return_trait_weights:
        meta = trait_weights
    else:
        meta = labels_relational

    if keeper_idxs is not None:
        data = np.array([data[idx] for idx in keeper_idxs])

    (train_data, test_data, train_labels, test_labels) = train_test_split(data, labels_reg,
                                                                          test_size=0.25,
                                                                          random_state=42)

    if K.image_data_format() == "channels_first":
        train_data = train_data.reshape((train_data.shape[0],
                                         input_shape[2], input_shape[1], input_shape[0]))
        test_data = test_data.reshape((test_data.shape[0],
                                       input_shape[2], input_shape[1], input_shape[0]))
    else:
        train_data = train_data.reshape((train_data.shape[0],
                                         input_shape[0], input_shape[1], input_shape[2]))
        test_data = test_data.reshape((test_data.shape[0],
                                       input_shape[0], input_shape[1], input_shape[2]))

    if normalize:
        train_data = train_data.astype("float32") / 255.0
        test_data = test_data.astype("float32") / 255.0
    if subtract_mean:
        if K.image_data_format() == "channels_first":
            tmp_data = train_data.reshape(train_data.shape[1], -1)
        else:
            tmp_data = train_data.reshape(train_data.shape[-1], -1)

        mean = np.mean(tmp_data, axis=1)
        train_data = train_data - mean
        test_data = test_data - mean

    if len(train_labels.shape) == 1 or train_labels.shape[1] == 1:
        le = LabelBinarizer()
        train_labels = le.fit_transform(train_labels)
        test_labels = le.transform(test_labels)
        target_names = [str(x) for x in le.classes_]
    else:
        target_names = []
    validation_data = (test_data, test_labels)
    train_data = (train_data, train_labels)

    if meta is not None:
        return train_data, validation_data, target_names, meta
    else:
        return train_data, validation_data, target_names


def configure_gpu_options():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    sess = Session(config=config)
    set_session(sess)


def get_command_line_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--checkpoints", type=str, default="checkpoints")
    ap.add_argument("-g", "--gpu", type=int, default=0)
    ap.add_argument("-s", "--smoothing", type=float, default=None,
                    help="amount of label smoothing to be applied")
    ap.add_argument("-p", "--params", type=str, default=None)
    ap.add_argument("-d", "--datapath", type=str, default='3dshapes.h5')
    ap.add_argument("-t", "--trait", type=str, default=None)
    ap.add_argument("-dw", "--dualweight", type=float, default=0.5)

    return vars(ap.parse_args())
