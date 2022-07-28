import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
from utils.load_data import collect_examples_per_class
from utils.config import get_cnn_paths, get_attribute_dict
import matplotlib.pyplot as plt
import matplotlib.path as mpath


def calc_tsne(key, components=2, verbose=1, perplexity=100, n_iter=2000, n_examples=100):
    """ calculates tsne embedding

    :param key: key to the pretrained cnn (bias + smoothing factor, e.g. 'default0-0')
    :param components: embedding dimension
    :param verbose: whether the calculation process is printed out
    :param perplexity: balances local and global aspects of the data (for details consult tsne paper)
    :param n_iter: iterations
    :param n_examples: number of examples per class used to calculate the embedding
    :return: 2D embedding of the vision module features
    """

    all_cnn_paths = get_cnn_paths()
    images, _ = collect_examples_per_class(n_examples=n_examples)

    network = tf.keras.models.load_model(all_cnn_paths[key])
    vision_module = tf.keras.Model(inputs=network.input, outputs=network.get_layer('dense_1').output)
    features = vision_module(images)
    tsne = TSNE(n_components=components, verbose=verbose, perplexity=perplexity, n_iter=n_iter)
    tsne_results = tsne.fit_transform(features)
    return tsne_results


def plot_tsne(tsne, points_per_class=50, title=None, alpha=0.2, n_examples=100):
    """ visualizes a 2D tsne embedding

    :param tsne: 2D embedding
    :param points_per_class: how many examples should be visualized per class
    :param title: plot title
    :param alpha: regulates transparency - opaqueness of the visualized data points
    :param n_examples: number of examples used to generate the embeddings
    :return: None (plot the embedding)
    """

    attribute_dict = get_attribute_dict()
    labels = []
    for i in range(64):
        labels += [i] * n_examples
    labels = np.array(labels)

    circle = mpath.Path.unit_circle()
    verts = np.copy(circle.vertices)
    verts[:, 1] *= 2

    colors = ['red', 'orange', 'teal', 'purple']
    sizes = [30, 70, 110, 150]
    shapes = ["s", "$\sqcap$", "o", 'd']

    plt.figure(figsize=(5, 5))

    for i in range(64):
        atts = attribute_dict[i]
        color = colors[np.argmax(atts[0:4])]
        size = sizes[np.argmax(atts[4:8])]
        shape = shapes[np.argmax(atts[8:12])]
        plt.scatter(tsne[labels == i, 0][0:points_per_class],
                    tsne[labels == i, 1][0:points_per_class],
                    color=color, marker=shape, s=size, alpha=alpha)
    if title:
        plt.title(title, fontsize=15)
