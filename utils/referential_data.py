import numpy as np
from utils.config import get_attribute_dict


def make_referential_data(data_set, n_distractors=2, return_permutation=False, irrelevant_attribute=None):
    """ function for generating the sender and receiver inputs in the reference game

    @param data_set: input data set: sender targets, receiver targets (images or feature vectors), labels
    @param n_distractors: number of distractors for the receiver (int)
    @param return_permutation: return the permutation indices of the data when translated to sender input (bool)
    @param irrelevant_attribute: provides irrelevant attribute, if existing (str)
                                    None --> all attributes are relevant
                                    'color' --> color is irrelevant
                                    'scale' --> scale is irrelevant
                                    'shape' --> shape is irrelevant
    """

    # generate indices for irrelevant attribute in symbolic encoding vector
    if irrelevant_attribute == 'color':
        irrelevant_attribute_indices = [0, 1, 2, 3]
    elif irrelevant_attribute == 'scale':
        irrelevant_attribute_indices = [4, 5, 6, 7]
    elif irrelevant_attribute == 'shape':
        irrelevant_attribute_indices = [8, 9, 10, 11]
    else:
        irrelevant_attribute_indices = []

    # the sender's and receiver's target images correspond to the images in the data set
    targets_sender, targets_receiver, target_labels = data_set
    target_labels_non_hot = np.argmax(target_labels, axis=1)

    n_data = len(target_labels)
    n_total = max(target_labels_non_hot) + 1
    classes = [c for c in range(n_total)]

    # generate a list of distractors for each target image
    distractors = [np.zeros_like(targets_receiver) for _ in range(n_distractors)]
    distractor_labels = [np.zeros_like(target_labels) for _ in range(n_distractors)]

    # if one attribute is irrelevant change the labels accordingly --> objects that are the same apart from this
    # attribute value have the same class label
    if len(irrelevant_attribute_indices) > 0:
        attributes = get_attribute_dict()
        relevant_attributes = np.zeros((len(attributes), len(attributes[0]) - len(irrelevant_attribute_indices)))
        for i in range(64):
            relevant_attributes[i] = np.delete(attributes[i], irrelevant_attribute_indices)
        label_dict = {}
        new_labels = []
        counter = -1
        for att in relevant_attributes:
            if str(att) in label_dict.keys():
                new_labels.append(label_dict[str(att)])
            else:
                counter = counter + 1
                new_labels.append(counter)
                label_dict[str(att)] = counter
        target_labels_non_hot = [new_labels[old_label] for old_label in target_labels_non_hot]
        classes = np.unique(new_labels)

    # make a list of possible distractor indices for each class label
    possible_distractor_indices = [np.where(target_labels_non_hot != i)[0] for i in classes]

    # shuffle the receiver targets for each object class, such that sender and receiver target class are the
    # same while the instances of these classes can be different (e.g. different wall color)
    targets_receiver_shuffled = np.zeros_like(targets_receiver)
    for cat in classes:
        targets_receiver_cat = targets_receiver[target_labels_non_hot == cat]
        np.random.shuffle(targets_receiver_cat)
        targets_receiver_shuffled[target_labels_non_hot == cat] = targets_receiver_cat

    # sample distractors using the distractor indices
    for j, l in enumerate(classes):
        n_samples = int(np.sum(target_labels_non_hot == l))
        random_indices = np.random.choice(possible_distractor_indices[j],
                                          size=n_samples * n_distractors,
                                          replace=True)
        random_indices = np.reshape(random_indices, (n_samples, n_distractors))

        for d in range(n_distractors):
            distractors[d][target_labels_non_hot == l] = targets_receiver[random_indices[:, d]]
            distractor_labels[d][target_labels_non_hot == l] = target_labels[random_indices[:, d]]

    # generate receiver input from targets and distractors
    receiver_input = np.stack([targets_receiver_shuffled] + distractors, axis=1)
    referential_labels = np.zeros((len(targets_sender), n_distractors + 1), dtype=np.float32)
    referential_labels[:, 0] = 1

    # permute targets and distractors in receiver input so they do not always follow the same order
    permutation = []
    for i in range(n_data):
        perm = np.random.permutation(n_distractors + 1)
        receiver_input[i] = receiver_input[i, perm]
        referential_labels[i] = referential_labels[i, perm]
        permutation.append(perm)

    target_and_distractor_labels = [target_labels] + distractor_labels

    if return_permutation:
        return targets_sender, receiver_input, referential_labels, target_and_distractor_labels, permutation
    else:
        return targets_sender, receiver_input, referential_labels, target_and_distractor_labels
