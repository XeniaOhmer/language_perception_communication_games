import numpy as np
from matplotlib import pyplot as plt
from utils.bootstrap_ci import bootstrap_mean
from mpl_toolkits.axes_grid1 import make_axes_locatable


def mixed_table(combinations, mode='language_emergence_basic', vs=4, ml=3, n_runs=10):
    """ generate the table of mean rewards withour symmeterizing the games: entries for 'bias1_bias2' will provide the
    value for a sender with bias1 playing with a receiver with bias2.

    :param combinations:    list of all possible bias combinations, where each entry is structured as follows:
                            'bias' - both agents have the same bias
                            'bias1_bias2' - agents have different biases
                            e.g. 'default_color' could be one combination
    :param mode: training setup (e.g. 'language_emergence_basic')
    :param vs: vocabulary size
    :param ml: message length
    :param n_runs: number of runs per combination
    :return: matrix with mean rewards per combination
    """

    if '-' in combinations[0]:
        mapping = {'color-scale': 0, 'color-shape': 1, 'scale-shape': 2}
    else:
        mapping = {'default': 0, 'color': 1, 'scale': 2, 'shape': 3, 'all': 4}

    rewards = {}

    for combination in combinations:
        rewards_combination = []
        for run in range(n_runs):
            reward = np.load('results/' + mode + '/' + combination + str(run) + '/vs' + str(vs)
                             + '_ml' + str(ml) + '/test_reward.npy')
            rewards_combination.append(reward[-1])
        rewards[combination] = np.mean(rewards_combination)

    table = np.zeros((len(mapping), len(mapping)))
    for key in rewards.keys():
        if '_' in key:
            key1, key2 = key.split('_')
            table[mapping[key1], mapping[key2]] = rewards[key]
        else:
            table[mapping[key], mapping[key]] = rewards[key]

    return table


def combined_table(combinations, mode='language_emergence_basic', vs=4, ml=3, n_runs=10):
    """ Here, a matrix with mean rewards is return that symmeterizes the game.
        Results for sender-bias1, receiver-bias2 and sender-bias2, receiver-bias1 are averaged.

    parameters: see mixed_table; number of runs here is given as for mixed table (do not double for symmeterization)
    :return: symmetric table with mean rewards
    """
    table = mixed_table(combinations, mode=mode, vs=vs, ml=ml, n_runs=n_runs)

    combined = (table + np.transpose(table)) / 2
    return combined


def show_table(combinations, table, vmin=0.9, vmax=1.0, upper=False, xlabel='bias agent 1', ylabel='bias agent 2',
               title='', cmap='viridis'):
    """ function for plotting the reward matrices

    :param combinations:    list of all possible bias combinations
    :param table: reward matrix
    :param vmin: color map minimum
    :param vmax: color map maximum
    :param upper: if true, show only upper matrix
    :param xlabel: xlabel
    :param ylabel: ylabel
    :param title: title
    :param cmap: color map
    :return: None (plot table
    """

    if '-' in combinations[0]:
        mapping = {'color-scale': 0, 'color-shape': 1, 'scale-shape': 2}
        labels = [k.split('-')[0] + '-\n' + k.split('-')[1] for k in mapping.keys()]
    else:
        mapping = {'default': 0, 'color': 1, 'scale': 2, 'shape': 3, 'all': 4}
        labels = mapping.keys()

    plt.figure(figsize=(len(mapping) + 1, len(mapping) + 1))
    if upper:
        mask = np.tri(table.shape[0], k=-1)
        table = np.ma.array(table, mask=mask)
    im = plt.imshow(table, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.xlabel(xlabel, fontsize=17)
    plt.ylabel(ylabel, fontsize=17)
    plt.xticks(ticks=range(len(mapping)), labels=labels, fontsize=16)
    plt.yticks(ticks=range(len(mapping)), labels=labels, fontsize=16)
    plt.title(title + '\n', fontsize=18)
    for (j, i), label in np.ndenumerate(table):
        if upper:
            if i >= j:
                plt.text(i, j, round(label, 3), ha='center', va='center', color='k', fontsize=17)
        else:
            plt.text(i, j, round(label, 3), ha='center', va='center', color='k', fontsize=17)
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(im, cax=cax, ticks=[vmin, 1.0])
    cbar.ax.set_ylabel('test reward', rotation=90, fontsize=17, labelpad=-10)
    cbar.ax.set_yticklabels([vmin, 1.0], fontsize=14)


def bootstrapped_cis(combinations, mode='language_emergence_basic', vs=4, ml=3, n_runs=10):
    """calculate bootstrapped confidence intervals for symmeterized matrices.

    :param combinations: list of all possible bias combinations
    :param mode: training setup (e.g. 'language_emergence_basic')
    :param vs: vocabulary size
    :param ml: message length
    :param n_runs: number of runs per combination (non-symmetric)
    :return: confidence intervals (absolute), means, confidence intervals (relative to mean)
    """
    if '-' in combinations[0]:
        mapping = {'color-scale': 0, 'color-shape': 1, 'scale-shape': 2}
        reverse_mapping = {0: 'color-scale', 1: 'color-shape', 2: 'scale-shape'}
    else:
        mapping = {'default': 0, 'color': 1, 'scale': 2, 'shape': 3, 'all': 4}
        reverse_mapping = {0: 'default', 1: 'color', 2: 'scale', 3: 'shape', 4: 'all'}

    rewards = np.zeros((len(mapping), len(mapping), n_runs))

    for combination in combinations:

        rewards_combination = []

        if '_' in combination:
            net1, net2 = combination.split('_')
        else:
            net1 = combination
            net2 = combination

        for run in range(n_runs):
            reward = np.load('results/' + mode + '/' + combination + str(run) + '/vs' + str(vs)
                             + '_ml' + str(ml) + '/test_reward.npy')
            rewards_combination.append(reward[-1])

        rewards[mapping[net1], mapping[net2], :] = rewards_combination

    folded = {}
    for i in range(len(mapping)):
        for j in range(i, len(mapping)):
            if i != j:
                folded[str(i) + str(j)] = np.concatenate((rewards[i, j, :], rewards[j, i, :]))
            else:
                folded[str(i) + str(j)] = rewards[i, j, :]
    cis = {}
    means = {}
    mean_intervals = {}
    for key in folded:
        ci = bootstrap_mean(folded[key])
        new_key = reverse_mapping[int(key[0])] + '_' + reverse_mapping[int(key[1])]
        cis[new_key] = (np.round(ci[0], 5), np.round(ci[1], 5))
        means[new_key] = np.mean(folded[key])
        mean_intervals[new_key] = (means[new_key] - cis[new_key][0], means[new_key] + cis[new_key][1])

    return mean_intervals, means, cis
