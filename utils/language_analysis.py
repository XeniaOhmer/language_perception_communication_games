import tensorflow as tf
import pickle
from nn import agents
from utils.entropy_scores import EntropyScores
from utils.config import get_cnn_paths
from utils.load_data import load_train
import warnings
warnings.filterwarnings("ignore")


def evaluate_entropy_scores(mode, conditions, vs, n_runs, n_epochs=150, cnns=None):
    """ evaluate the entropy scores: effectiveness, efficiency, positional disentanglement,
    bag-of-symbol disentanglement, residual entropy

    :param mode: training mode (folder in results)
    :param conditions: list of bias conditions, e.g. 'default' (subfolder in results)
    :param vs: vocab size
    :param n_runs: number of runs per condition
    :param n_epochs: number of epochs
    :param cnns: cnn keys for the sender vision module
    :return: None (save entropy scores in the respective results folder)
    """

    data, labels = load_train()
    all_cnn_paths = get_cnn_paths()

    for c, condition in enumerate(conditions):
        print('calculating condition: ' + str(condition))
        for run in range(n_runs):
            path = 'results/' + mode + '/' + condition + str(run) + '/vs' + str(vs) + '_ml3/'
            sender = agents.Sender(vs, 3, 128, 128, None)
            cnn_sender = tf.keras.models.load_model(all_cnn_paths[cnns[c]])
            cnn_sender = tf.keras.Model(inputs=cnn_sender.input,
                                        outputs=cnn_sender.get_layer('dense_1').output)
            inputs = cnn_sender.predict(data)
            sender.load_weights(path + 'sender_weights_epoch' + str(n_epochs - 1) + '/')
            messages, _, _, _, _ = sender.forward(inputs, training=False)
            messages = messages.numpy()
            entropy_scores = EntropyScores(messages, labels)
            all_scores = entropy_scores.calc_all_scores()
            pickle.dump(all_scores, open(path + 'entropy_scores.pkl', 'wb'))
