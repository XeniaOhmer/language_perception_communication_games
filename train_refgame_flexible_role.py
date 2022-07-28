import tensorflow as tf
from tensorflow.keras import models
from utils.referential_data import *
from utils.config import *
from nn.flexible_role_agents import FlexibleRoleAgent, ClassificationFlexibleRoleAgent
from nn.flexible_role_training import FlexibleRoleTrainer
import os
import pickle
import argparse
import logging

parser = argparse.ArgumentParser()

parser.add_argument("--activation", type=str, default='tanh',
                    help='activation function of the layer mapping visual representations to hidden state dimension')
parser.add_argument("--hidden_dim", type=int, default=128, help='language module hidden layer dimension')
parser.add_argument("--embed_dim", type=int, default=128, help='language module embedding layer dimension')
parser.add_argument("--vocab_size", type=int, default=4, help='vocabulary size of the agents, i.e. number of symbols')
parser.add_argument("--message_length", type=int, default=3, help='(maximal) length of the messages')
parser.add_argument("--entropy_coeff_sender", type=float, default=0.02, help='entropy regularization term, sender')
parser.add_argument("--entropy_coeff_receiver", type=float, default=0.02, help='entropy regularization term, receiver')
parser.add_argument("--entropy_annealing", type=float, default=1.,
                    help='annealing schedule for the entropy regularization, 1 means no annealing')
parser.add_argument("--n_epochs", type=int, default=150, help='number of training epochs')
parser.add_argument("--batch_size", type=int, default=128, help='batch size')
parser.add_argument("--sim_agent1", type=str, default='default', help='bias condition agent 1')
parser.add_argument("--sim_agent2", type=str, default='default', help='bias condition agent 2')
parser.add_argument("--sf_agent1", type=str, default='0-0', help='smoothing factor agent 1')
parser.add_argument("--sf_agent2", type=str, default='0-0', help='smoothing factor agent 2')
parser.add_argument("--run", type=str, default='default_default', help='name of the run')
parser.add_argument("--learning_rate", type=float, default=0.0005, help='learning rate for training')
parser.add_argument("--n_distractors", type=int, default=2, help='number of distractors for receiver')
parser.add_argument("--mode", type=str, default='test_FR', help='name of results folder')
parser.add_argument("--train_vision", type=bool, default=False, help='whether to train sender vision module')
parser.add_argument("--classification", type=bool, default=False, help='whether classification is also trained')
parser.add_argument("--load_sender_epoch", type=int, default=None, help='if sender is fixed, which epoch to load')
parser.add_argument("--flexible_message_length", type=bool, default=False,
                    help='whether message length is fixed or flexible')
parser.add_argument("--length_cost", type=float, default=0.0,
                    help='additional cost term for producing longer messages, only makes sense if length is flexible')
parser.add_argument("--irrelevant_attribute", type=str, default=None, help='indicate if an attribute is irrelevant')
parser.add_argument("--n_runs", type=int, default=10, help='number of runs for this simulation')
args = parser.parse_args()

effective_vocab_size = args.vocab_size

for r in range(args.n_runs):

    all_cnn_paths, image_dim, n_classes, feature_dims = get_config()

    # stores all results in a folder 'results/'
    path = ('results/' + str(args.mode) + '/' + str(args.run) + str(r) + '/' +
            'vs' + str(args.vocab_size) + '_ml' + str(args.message_length) + '/')

    assert args.irrelevant_attribute is None or not args.classification, \
        "classification not implemented for grouping of attributes"

    # store parameters for each run in respective subfolder

    params = {"batch_size": args.batch_size,
              "similarity_agent1": args.sim_agent1,
              "similarity_agent2": args.sim_agent2,
              "smoothing_factor_agent1": args.sf_agent1,
              "smoothing_factor_agent2": args.sf_agent2,
              "hidden_dim": args.hidden_dim,
              "embed_dim": args.embed_dim,
              "vocab_size": args.vocab_size,
              "message_length": args.message_length,
              "entropy_coeff_sender": args.entropy_coeff_sender,
              "entropy_coeff_receiver": args.entropy_coeff_receiver,
              "flexible_message_length": False,
              "length_cost": args.length_cost,
              "learning_rate": args.learning_rate,
              "n_distractors": args.n_distractors,
              "activation": args.activation,
              "entropy_annealing": args.entropy_annealing,
              "train_vision": args.train_vision,
              "classification": args.classification,
              "irrelevant_feature": args.irrelevant_attribute,
              "fixed_sender": False,
              }

    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + 'params.pkl', 'wb') as handle:
        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # log training progress in a text file

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(filename=path + "log_file.txt",
                        level=logging.DEBUG,
                        format='%(levelname)s: %(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S')

    # load pretrained models

    cnn_path_1 = all_cnn_paths[args.sim_agent1 + args.sf_agent1]
    cnn_1 = models.load_model(cnn_path_1)
    vision_module_1 = tf.keras.Model(inputs=cnn_1.input,
                                     outputs=cnn_1.get_layer('dense_1').output)

    cnn_path_2 = all_cnn_paths[args.sim_agent2 + args.sf_agent2]
    cnn_2 = models.load_model(cnn_path_2)
    vision_module_2 = tf.keras.Model(inputs=cnn_2.input,
                                     outputs=cnn_2.get_layer('dense_1').output)

    # initialize sender, receiver and training classes

    agents = []
    if args.classification and args.train_vision:

        for i in range(2):
            cnn = [cnn_1, cnn_2][i]
            vision_module = [vision_module_1, vision_module_2][i]
            classification_module = tf.keras.Sequential([cnn.get_layer('dense_2')])

            agents.append(
                ClassificationFlexibleRoleAgent(
                    effective_vocab_size,
                    args.message_length,
                    args.embed_dim,
                    args.hidden_dim,
                    vision_module,
                    classification_module,
                    n_distractors=args.n_distractors,
                    image_dim=image_dim,
                    activation=args.activation,
                    train_vision=args.train_vision)
            )

    else:
        for i in range(2):
            if args.train_vision:
                vision_module = [vision_module_1, vision_module_2][i]
            else:
                vision_module = None

            agents.append(
                FlexibleRoleAgent(
                    effective_vocab_size,
                    args.message_length,
                    args.embed_dim,
                    args.hidden_dim,
                    vision_module,
                    n_distractors=args.n_distractors,
                    image_dim=image_dim,
                    activation=args.activation,
                    train_vision=args.train_vision)
            )

    trainer = FlexibleRoleTrainer(agents,
                                  lr=args.learning_rate,
                                  entropy_coeff_sender=args.entropy_coeff_sender,
                                  entropy_coeff_receiver=args.entropy_coeff_receiver,
                                  classification=args.classification,
                                  train_vision=args.train_vision,
                                  )

    # training

    all_reward = []
    all_length = []
    all_agent1_sender_loss = []
    all_agent1_receiver_loss = []
    all_agent2_sender_loss = []
    all_agent2_receiver_loss = []
    all_test_reward = []
    if args.classification:
        all_classification_accurracies = []

    train_data = np.load('3Dshapes_subset/train_data.npy')
    train_labels = np.load('3Dshapes_subset/train_labels.npy')
    test_data = np.load('3Dshapes_subset/test_data.npy')
    test_labels = np.load('3Dshapes_subset/test_labels.npy')
    a = np.argmax(train_labels, axis=1)
    b = np.argmax(test_labels, axis=1)

    n_samples = len(train_data)

    if args.train_vision:
        both_train = [train_data, train_data]
        both_test = [test_data, test_data]

    else:
        both_train = []
        both_test = []
        for i in range(2):
            both_train.append([vision_module_1, vision_module_2][i].predict(train_data))
            both_test.append([vision_module_1, vision_module_2][i].predict(test_data))

    del train_data

    for epoch in range(args.n_epochs):

        # prepare dataset for referential game from shape data

        referential_data1 = make_referential_data(
            [both_train[0], both_train[1], train_labels],
            n_distractors=args.n_distractors,
            irrelevant_attribute=args.irrelevant_attribute
        )

        if args.train_vision:
            train_dataset = tf.data.Dataset.from_tensor_slices((referential_data1[0],
                                                                referential_data1[1],
                                                                referential_data1[2]))
            if args.classification:
                agent1_labels, agent2_labels = get_smoothed_labels(
                    referential_data1[3][0], args.sim_agent1, args.sim_agent2, args.sf_agent1, args.sf_agent2
                )

                train_dataset = tf.data.Dataset.from_tensor_slices((referential_data1[0],
                                                                    referential_data1[1],
                                                                    referential_data1[2],
                                                                    agent1_labels,
                                                                    agent2_labels))
        else:
            referential_data2 = make_referential_data(
                [both_train[1], both_train[0], train_labels],
                n_distractors=args.n_distractors,
                irrelevant_attribute=args.irrelevant_attribute
            )
            train_dataset = tf.data.Dataset.from_tensor_slices((referential_data1[0],
                                                                referential_data1[1],
                                                                referential_data1[2],
                                                                referential_data2[0],
                                                                referential_data2[1],
                                                                referential_data2[2]))

        train_dataset = train_dataset.shuffle(buffer_size=5000)
        train_dataset = train_dataset.batch(args.batch_size)

        rewards, mean_length, sender_loss, receiver_loss = trainer.train_epoch(train_dataset)

        trainer.entropy_coeff_receiver = trainer.entropy_coeff_receiver * args.entropy_annealing
        trainer.entropy_coeff_sender = trainer.entropy_coeff_sender * args.entropy_annealing

        del referential_data1
        if not args.train_vision:
            del referential_data2
        del train_dataset

        all_reward.append(rewards)
        all_length.append(mean_length)
        all_agent1_sender_loss.append(sender_loss[0])
        all_agent1_receiver_loss.append(receiver_loss[0])
        all_agent2_sender_loss.append(sender_loss[1])
        all_agent2_receiver_loss.append(receiver_loss[1])

        # get test data and messages

        referential_data1 = make_referential_data(
            [both_test[0], both_test[1], test_labels],
            n_distractors=args.n_distractors,
            irrelevant_attribute=args.irrelevant_attribute
        )

        if args.train_vision:
            test_dataset = tf.data.Dataset.from_tensor_slices((referential_data1[0],
                                                               referential_data1[1],
                                                               referential_data1[2]))
        else:
            referential_data2 = make_referential_data(
                [both_test[1], both_test[0], test_labels],
                n_distractors=args.n_distractors,
                irrelevant_attribute=args.irrelevant_attribute
            )
            test_dataset = tf.data.Dataset.from_tensor_slices((referential_data1[0],
                                                               referential_data1[1],
                                                               referential_data1[2],
                                                               referential_data2[0],
                                                               referential_data2[1],
                                                               referential_data2[2]))

        # evaluate test rewards
        test_dataset = test_dataset.batch(args.batch_size)
        test_reward = trainer.evaluate(test_dataset)
        all_test_reward.append(test_reward)
        del test_dataset

        if args.classification:
            test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
            test_dataset = test_dataset.batch(args.batch_size)
            class_acc = trainer.evaluate_classification(test_dataset)
            all_classification_accurracies.append(class_acc)
            del test_dataset
            logging.info("epoch {0}, rewards train {1}, rewards test {2}, acc {3}".format(
                epoch, rewards, test_reward, class_acc))
        else:
            logging.info(
                "epoch {0}, rewards train {1}, rewards test {2}".format(epoch, rewards, test_reward))

        # save sender and receiver weights

        if epoch == args.n_epochs - 1:
            trainer.agents[0].save_weights(path + 'agent1_weights_epoch' + str(epoch) + '/')
            trainer.agents[1].save_weights(path + 'agent2_weights_epoch' + str(epoch) + '/')

    # save training results

    np.save(path + 'reward.npy', all_reward)
    np.save(path + 'length.npy', all_length)
    np.save(path + 'agent1_sender_loss.npy', all_agent1_sender_loss)
    np.save(path + 'agent1_receiver_loss.npy', all_agent1_receiver_loss)
    np.save(path + 'agent2_sender_loss.npy', all_agent2_sender_loss)
    np.save(path + 'agent2_receiver_loss.npy', all_agent2_receiver_loss)
    np.save(path + 'test_reward.npy', all_test_reward)
    if args.classification:
        np.save(path + 'classification_acc.npy', all_classification_accurracies)
