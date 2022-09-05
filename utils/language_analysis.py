import pickle
from utils.entropy_scores import EntropyScores
from utils.load_data import load_train
import warnings

warnings.filterwarnings("ignore")


def calculate_and_save_entropy_scores(path, senders, cnn_sender, flexible_role=False):
    """ evaluate the entropy scores: effectiveness, efficiency, positional disentanglement,
    bag-of-symbol disentanglement, residual entropy

    :param path: path for storing the results
    :param senders: list of sender agents
    :param cnn_sender:  original vision module of the senders (before training),
                        either a single keras model, or a list of keras models in case of different original modules
    :param flexible_role: whether agents are flexible-role agents

    :return: None (save entropy scores in the respective results folder)
    """

    print('calculating entropy scores')

    data, labels = load_train()

    for i in range(len(senders)):

        print('sender', i)
        appendix = '' if len(senders) == 1 else str(i)

        if senders[i].vision_module:
            inputs = data
        else:
            if isinstance(cnn_sender, list):
                cnn = cnn_sender[i]
            else:
                cnn = cnn_sender
            inputs = cnn.predict(data)

        if not flexible_role:
            messages, _, _, _, _ = senders[i].forward(inputs, training=False)
        else:
            messages, _, _, _, _ = senders[i].sender_forward(inputs, training=False)

        messages = messages.numpy()
        entropy_scores = EntropyScores(messages, labels)
        all_scores = entropy_scores.calc_all_scores()
        print(all_scores)
        pickle.dump(all_scores, open(path + 'entropy_scores' + appendix + '.pkl', 'wb'))

