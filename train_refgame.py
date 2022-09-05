import tensorflow as tf
from tensorflow.keras import models
from utils.referential_data import *
from utils.config import *
from utils.train import load_data
from nn.training import Trainer
from nn.agents import Sender, Receiver, ClassificationSender, ClassificationReceiver
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
parser.add_argument("--n_senders", type=int, default=1, help='number of senders')
parser.add_argument("--n_receivers", type=int, default=1, help='number of receivers')
parser.add_argument("--n_epochs", type=int, default=150, help='number of training epochs')
parser.add_argument("--batch_size", type=int, default=128, help='batch size')
parser.add_argument("--sim_sender", type=str, default='default', help='bias condition sender')
parser.add_argument("--sim_receiver", type=str, default='default', help='bias condition receiver')
parser.add_argument("--sf_sender", type=str, default='0-0', help='smoothing factor sender')
parser.add_argument("--sf_receiver", type=str, default='0-0', help='smoothing factor receiver')
parser.add_argument("--run", type=str, default='test', help='name of the run')
parser.add_argument("--learning_rate", type=float, default=0.0005, help='learning rate for training')
parser.add_argument("--n_distractors", type=int, default=2, help='number of distractors for receiver')
parser.add_argument("--mode", type=str, default='test', help='name of results folder')
parser.add_argument("--train_vision_sender", type=bool, default=False, help='whether to train sender vision module')
parser.add_argument("--train_vision_receiver", type=bool, default=False, help='whether to train receiver vision module')
parser.add_argument("--classification", type=bool, default=False, help='whether classification is also trained')
parser.add_argument("--load_sender_from", type=str, default=None, help='if sender is fixed, subdir with sender weights')
parser.add_argument("--load_sender_epoch", type=int, default=150, help='which epoch to load for fixed sender')
parser.add_argument("--flexible_message_length", type=bool, default=False,
                    help='whether message length is fixed or flexible')
parser.add_argument("--length_cost", type=float, default=0.0,
                    help='additional cost term for producing longer messages, only makes sense if length is flexible')
parser.add_argument("--irrelevant_attribute", type=str, default=None, help='indicate if an attribute is irrelevant')
parser.add_argument("--n_runs", type=int, default=10, help='number of runs for this simulation')
parser.add_argument("--analyze_language", type=bool, default=False, help="whether to calculate and store entropy scores")
parser.add_argument("--analyze_vision", type=bool, default=False, help="whether to calculate and store rsa scores")
parser.add_argument("--save_weights", type=bool, default=False, help="whether to save agents' weights")
args = parser.parse_args()

if args.flexible_message_length:
    effective_vocab_size = args.vocab_size + 1
else:
    effective_vocab_size = args.vocab_size

if args.load_sender_from is not None:
    sender_fixed = True
    sender_epoch = args.load_sender_epoch - 1
else:
    sender_fixed = False


for r in range(args.n_runs):

    all_cnn_paths, image_dim, n_classes, feature_dims = get_config()

    # stores all results in a folder 'results/'
    path = ('results/' + str(args.mode) + '/' + str(args.run) + str(r) + '/' +
            'vs' + str(args.vocab_size) + '_ml' + str(args.message_length) + '/')

    assert args.irrelevant_attribute is None or not args.classification, \
        "classification not implemented for grouping of attributes"

    # store parameters for each run in respective subfolder

    params = {"batch_size": args.batch_size,
              "similarity_sender": args.sim_sender,
              "similarity_receiver": args.sim_receiver,
              "smoothing_factor_sender": args.sf_sender,
              "smoothing_factor_receiver": args.sf_receiver,
              "hidden_dim": args.hidden_dim,
              "embed_dim": args.embed_dim,
              "vocab_size": args.vocab_size,
              "message_length": args.message_length,
              "entropy_coeff_sender": args.entropy_coeff_sender,
              "entropy_coeff_receiver": args.entropy_coeff_receiver,
              "flexible_message_length": args.flexible_message_length,
              "length_cost": args.length_cost,
              "learning_rate": args.learning_rate,
              "n_distractors": args.n_distractors,
              "activation": args.activation,
              "entropy_annealing": args.entropy_annealing,
              "train_vision_sender": args.train_vision_sender,
              "train_vision_receiver": args.train_vision_receiver,
              "classification": args.classification,
              "irrelevant_feature": args.irrelevant_attribute,
              "fixed_sender": sender_fixed,
              "n_senders": args.n_senders,
              "n_receivers": args.n_receivers
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

    cnn_path_sender = all_cnn_paths[args.sim_sender + args.sf_sender]
    cnn_path_receiver = all_cnn_paths[args.sim_receiver + args.sf_receiver]

    vision_modules_sender = []
    for i in range(args.n_senders):
        cnn_sender = models.load_model(cnn_path_sender)
        vision_modules_sender.append(tf.keras.Model(inputs=cnn_sender.input,
                                                    outputs=cnn_sender.get_layer('dense_1').output))
    vision_modules_receiver = []
    for i in range(args.n_receivers):
        cnn_receiver = models.load_model(cnn_path_receiver)
        vision_modules_receiver.append(tf.keras.Model(inputs=cnn_receiver.input,
                                                      outputs=cnn_receiver.get_layer('dense_1').output))

    # initialize sender, receiver and training classes
    senders = []
    receivers = []

    for i in range(args.n_senders):
        if args.classification and args.train_vision_sender:
            classification_module_sender = tf.keras.Sequential([cnn_sender.get_layer('dense_2')])
            sender = ClassificationSender(
                effective_vocab_size,
                args.message_length,
                args.embed_dim,
                args.hidden_dim,
                vision_modules_sender[i],
                classification_module_sender,
                flexible_message_length=args.flexible_message_length,
                activation=args.activation,
                train_vision=args.train_vision_sender)
        else:
            vision_module = vision_modules_sender[i] if args.train_vision_sender else None
            sender = Sender(
                effective_vocab_size,
                args.message_length,
                args.embed_dim,
                args.hidden_dim,
                vision_module,
                flexible_message_length=args.flexible_message_length,
                activation=args.activation,
                train_vision=args.train_vision_sender)
            if sender_fixed:
                sender_path = ('results/' + args.load_sender_from + args.sim_sender + str(r) + '/vs'
                               + str(args.vocab_size) + '_ml' + str(args.message_length) + '/')
                if args.n_senders > 1:
                    sender.load_weights(sender_path + 'sender' + str(i) + '_weights_epoch' + str(sender_epoch) + '/')
                else:
                    sender.load_weights(sender_path + 'sender_weights_epoch' + str(sender_epoch) + '/')
        senders.append(sender)

    for i in range(args.n_receivers):
        if args.classification and args.train_vision_receiver:
            classification_module_receiver = tf.keras.Sequential([cnn_receiver.get_layer('dense_2')])
            receiver = ClassificationReceiver(
                effective_vocab_size,
                args.message_length,
                args.embed_dim,
                args.hidden_dim,
                vision_modules_receiver[i],
                classification_module_receiver,
                flexible_message_length=args.flexible_message_length,
                activation=args.activation,
                image_dim=image_dim,
                n_distractors=args.n_distractors,
                train_vision=args.train_vision_receiver)
        else:
            vision_module = vision_modules_receiver[i] if args.train_vision_receiver else None
            receiver = Receiver(
                effective_vocab_size,
                args.message_length,
                args.embed_dim,
                args.hidden_dim,
                vision_module,
                flexible_message_length=args.flexible_message_length,
                activation=args.activation,
                n_distractors=args.n_distractors,
                image_dim=image_dim,
                train_vision=args.train_vision_receiver)
        receivers.append(receiver)

    trainer = Trainer(senders,
                      receivers,
                      entropy_coeff_sender=args.entropy_coeff_sender,
                      entropy_coeff_receiver=args.entropy_coeff_receiver,
                      length_cost=args.length_cost,
                      sender_lr=args.learning_rate,
                      receiver_lr=args.learning_rate,
                      classification=args.classification,
                      train_vision_sender=args.train_vision_sender,
                      train_vision_receiver=args.train_vision_receiver,
                      sender_fixed=sender_fixed)

    # training

    all_reward = []
    all_length = []
    all_sender_loss = []
    all_receiver_loss = []
    all_test_reward = []
    if args.classification:
        all_classification_accurracies = []

    (train_data, train_labels), (test_data, test_labels), _, _ = load_data((64, 64, 3))

    n_samples = len(train_data)

    if not args.train_vision_sender:
        sender_train = vision_modules_sender[0].predict(train_data)
        sender_test = vision_modules_sender[0].predict(test_data)
    else:
        sender_train = train_data
        sender_test = test_data
    if not args.train_vision_receiver:
        receiver_train = vision_modules_receiver[0].predict(train_data)
        receiver_test = vision_modules_receiver[0].predict(test_data)
    else:
        receiver_train = train_data
        receiver_test = test_data

    del train_data

    for epoch in range(args.n_epochs):

        # prepare dataset for referential game from shape data

        sender_in, receiver_in, referential_labels, td_labels = make_referential_data(
            [sender_train, receiver_train, train_labels],
            n_distractors=args.n_distractors,
            irrelevant_attribute=args.irrelevant_attribute
        )

        if not args.classification:
            train_dataset = tf.data.Dataset.from_tensor_slices((sender_in, receiver_in, referential_labels))
        else:
            sender_labels, receiver_labels = get_smoothed_labels(
                td_labels[0], args.sim_sender, args.sim_receiver, args.sf_sender, args.sf_receiver
            )
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (sender_in, receiver_in, referential_labels, sender_labels, receiver_labels)
            )
        train_dataset = train_dataset.shuffle(buffer_size=5000)
        train_dataset = train_dataset.batch(args.batch_size)

        rewards, mean_length, sender_loss, receiver_loss = trainer.train_epoch(train_dataset)

        trainer.entropy_coeff_receiver = trainer.entropy_coeff_receiver * args.entropy_annealing
        trainer.entropy_coeff_sender = trainer.entropy_coeff_sender * args.entropy_annealing

        del sender_in
        del receiver_in
        del train_dataset

        all_reward.append(rewards)
        all_length.append(mean_length)
        all_sender_loss.append(sender_loss)
        all_receiver_loss.append(receiver_loss)

        # get test data and messages

        sender_in, receiver_in, referential_labels, target_distractor_labels = make_referential_data(
            [sender_test, receiver_test, test_labels],
            n_distractors=args.n_distractors,
            irrelevant_attribute=args.irrelevant_attribute
        )

        # evaluate test rewards

        test_dataset = tf.data.Dataset.from_tensor_slices((sender_in, receiver_in, referential_labels))
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

        if args.save_weights and epoch == args.n_epochs - 1:
            for i in range(args.n_senders):
                if args.n_senders > 1:
                    appendix = str(i)
                else:
                    appendix = ''
                trainer.senders[i].save_weights(path + 'sender' + appendix + '_weights_epoch' + str(epoch) + '/')
            for i in range(args.n_receivers):
                if args.n_receivers > 1:
                    appendix = str(i)
                else:
                    appendix = ''
                trainer.receivers[i].save_weights(path + 'receiver' + appendix + '_weights_epoch' + str(epoch) + '/')

    # save training results

    np.save(path + 'reward.npy', all_reward)
    np.save(path + 'length.npy', all_length)
    np.save(path + 'sender_loss.npy', all_sender_loss)
    np.save(path + 'receiver_loss.npy', all_receiver_loss)
    np.save(path + 'test_reward.npy', all_test_reward)
    if args.classification:
        np.save(path + 'classification_acc.npy', all_classification_accurracies)

    # analyze vision

    if args.analyze_vision:

        from utils.vision_analysis import *
        if sender_fixed:
            agent_type = 'receiver'
        else:
            agent_type = 'both'
        cnn_sender = models.load_model(cnn_path_sender)
        cnn_sender = tf.keras.Model(inputs=cnn_sender.input, outputs=cnn_sender.get_layer('dense_1').output)
        cnn_receiver = models.load_model(cnn_path_receiver)
        cnn_receiver = tf.keras.Model(inputs=cnn_receiver.input, outputs=cnn_receiver.get_layer('dense_1').output)
        calculate_and_save_rsa_scores(path, trainer.senders, trainer.receivers, cnn_sender, cnn_receiver,
                                      agent_type=agent_type, n_examples=50)

    # analyze language

    if args.analyze_language:

        from utils.language_analysis import *
        cnn_sender = models.load_model(cnn_path_sender)
        cnn_sender = tf.keras.Model(inputs=cnn_sender.input, outputs=cnn_sender.get_layer('dense_1').output)
        calculate_and_save_entropy_scores(path, trainer.senders, cnn_sender)

