from seaborn.algorithms import bootstrap
import numpy as np


def bootstrap_mean_difference(x, y, n=10000, ci=95):
    """ calculates the bootstrapped confidence interval for the differences in means of x and y.

    :param x: one dataset (1D array)
    :param y: another dataset (1D array)
    :param n: number of iterations
    :param ci: percentage of confidence interval (ci=95 --> 95% confidence interval)
    :return: lower and upper bound of the confidence interval
    """
    
    means_x = bootstrap(x, n_boot=n, func=np.mean)
    means_y = bootstrap(y, n_boot=n, func=np.mean)
    lim = (100-ci) / 2
    lower_bound = np.percentile(means_x-means_y, lim)
    upper_bound = np.percentile(means_x-means_y, 100-lim)
    
    return lower_bound, upper_bound


def bootstrap_mean(x, n=10000, ci=95):
    """ bootstraps the confidence interval for the mean of data x

    :param x: data (1D array)
    :param n: number of iterations
    :param ci: percentage of confidence interval (ci=95 --> 95% confidence interval)
    :return: lower bound and upper bound of the confidence interval
    """
    
    means = bootstrap(x, n_boot=n, func=np.mean)
    lim = (100-ci) / 2
    lower_bound = np.percentile(means, lim)
    upper_bound = np.percentile(means, 100-lim)
    
    return np.mean(means)-lower_bound, upper_bound-np.mean(means)
