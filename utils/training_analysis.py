import matplotlib.pyplot as plt
import numpy as np
from utils.bootstrap_ci import bootstrap_mean, bootstrap_mean_difference


def get_stats(name, vocab_size, message_length, mode, n_runs=10):
    """ collect train and test rewards, as well as classification accuracies if vision modules are also trained.

    :param name: name of the file (e.g. 'default')
    :param vocab_size: vocabulary size
    :param message_length: message length
    :param mode: training mode (result directors, e.g. 'language_emergence_basic')
    :param n_runs: number of runs
    :return:
    """
    rewards = []
    test_rewards = []
    class_acc = []

    for run in range(n_runs):
        filename = 'results/' + mode + '/' + name + str(run) + '/vs' + str(vocab_size) + '_ml' + str(message_length)

        rewards.append(np.load(filename + '/reward.npy'))
        test_rewards.append(np.load(filename + '/test_reward.npy'))

        if 'train_vision' in mode and not 'no_classification' in mode and not 'unbiased' in mode:
            class_acc.append(np.load(filename + '/classification_acc.npy'))

    return rewards, test_rewards, class_acc


def show_results_multiples(names, vocab_size, message_length, ylim=[0.8, 1.01], mode='language_emergence_basic',
                           subplots=(1, 4), n_runs=10, figsize=(15, 3.5)):
    """ plot the results for multiple runs of the same simulation

    :param names: names of the files
    :param vocab_size: vocabulary size
    :param message_length: message length
    :param ylim: y axis limits
    :param mode: training mode (e.g. 'language_emergence_basic')
    :param subplots: subplot structure
    :param n_runs: number of runs per name
    :param figsize: figure size
    :return: None (plot)
    """
    fig = plt.figure(figsize=figsize)
    for plot_index, name in enumerate(names):

        plt.subplot(subplots[0], subplots[1], plot_index + 1)

        rewards, test_rewards, class_acc = get_stats(name, vocab_size, message_length, mode, n_runs=n_runs)

        mean_R_train = np.mean(rewards, axis=0)
        max_R_train = np.max(rewards, axis=0)
        min_R_train = np.min(rewards, axis=0)

        plt.plot(mean_R_train, color='k')
        plt.plot(min_R_train, color='b', alpha=0.3)
        plt.plot(max_R_train, color='r', alpha=0.3)
        plt.fill_between(range(len(min_R_train)), min_R_train, y2=mean_R_train, color='blue', alpha=.1)
        plt.fill_between(range(len(max_R_train)), mean_R_train, y2=max_R_train, color='red', alpha=.1)

        plt.ylim(ylim)
        final_R_train = mean_R_train[-1]
        final_R_test = np.mean([R for R in test_rewards], axis=0)[-1]

        if '/' in name:
            name = name.partition('/')[0]

        title = str(name) + '\ntrain: ' + str(round(final_R_train, 3)) + ', test: ' + str(round(final_R_test, 3))

        if len(class_acc) > 0:
            try:
                title = title + '\nclass: ' + str(round(np.mean([class_acc[i][-1] for i in range(n_runs)]), 3))
            except:
                title = title + '\nclass: ' + str(np.NaN)

        plt.title(title)
        plt.xlabel('epoch')
        plt.ylabel('reward')

    fig.legend(labels=['mean', 'min', 'max'],  # The labels for each line
               loc="lower center",  # Position of legend
               borderaxespad=-0.3,  # Small spacing around legend box
               ncol=n_runs)

    fig.tight_layout()


def show_rewards(names, vocab_size, message_length, mode='language_emergence_basic', n_runs=10):
    """ print mean accuracies and bootstrapped confidence intervals across a number of runs

    :param names: names of the files
    :param vocab_size: vocabulary size
    :param message_length: message length
    :param mode: training mode (e.g. 'language_emergence_basic')
    :param n_runs: number of runs per name
    :return: prints mean train and test rewards across runs together with bootstrapped confidence intervals
    """
    for name in names:

        print('\n' + name)

        rewards, test_rewards, class_acc = get_stats(name, vocab_size, message_length, mode, n_runs=n_runs)

        mean_R_train = np.mean([R[-1] for R in rewards])
        mean_R_test = np.mean([R[-1] for R in test_rewards])
        bci_train = bootstrap_mean(np.array([R[-1] for R in rewards]))
        bci_test = bootstrap_mean(np.array([R[-1] for R in test_rewards]))

        print('train reward: ', round(mean_R_train, 3), 'bci', bci_train)
        print('test reward: ', round(mean_R_test, 3), 'bci', bci_test)

        if len(class_acc) > 0:
            mean_acc_class = np.mean([acc[-1] for acc in class_acc])
            bci_acc_class = bootstrap_mean(np.array([acc[-1] for acc in class_acc]))
            print('class acc', round(mean_acc_class, 3), 'bci', bci_acc_class)


def bootstrap_tests(name1, name2, vocab_size=4, message_length=3, mode='language_emergence_basic', n_runs=10):
    """ calculate the bootstrapped confidence interval for the difference in means for train and test rewards
    for two different simulations

    :param name1: file name for simulation 1 (e.g. 'default')
    :param name2: file name for simulation 2 (e.g. 'all')
    :param vocab_size: vocabulary size
    :param message_length: message length
    :param mode: training mode (e.g. 'language_emergence_basic')
    :param n_runs: number of runs per name
    :return: print the bootstrap confidence intervals for train and test rewards
    """
    decimals = 3
    R1, testR1, _ = get_stats(name1, vocab_size, message_length, mode, n_runs=n_runs)
    R2, testR2, _ = get_stats(name2, vocab_size, message_length, mode, n_runs=n_runs)

    # rewards

    train_result = bootstrap_mean_difference([r[-1] for r in R1], [r[-1] for r in R2])
    test_result = bootstrap_mean_difference([r[-1] for r in testR1], [r[-1] for r in testR2])

    print("bootstrap ci, values rounded to " + str(decimals) + " decimals", "\n")
    print("train reward", round(train_result[0], decimals), round(train_result[1], decimals))
    print("test reward", round(test_result[0], decimals), round(test_result[1], decimals))


