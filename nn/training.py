import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers


class Trainer:

    def __init__(self, senders, receivers, sender_lr=0.001, receiver_lr=0.001, length_cost=0.0,
                 entropy_coeff_sender=0.0, entropy_coeff_receiver=0.0, classification=False,
                 train_vision_sender=False, train_vision_receiver=False, sender_fixed=False):
        """ Constructor.

        :param senders: list of sender agents
        :param receivers: list of receiver agents
        :param sender_lr: learning rate sender
        :param receiver_lr: learning rate receiver
        :param length_cost: message length cost (float)
        :param entropy_coeff_sender: weight of entropy regularization for sender (float)
        :param entropy_coeff_receiver: weight of entropy regularization for receiver (float)
        :param classification: whether (at least one of) the agents are trained on the classification task (bool)
        :param train_vision_sender: whether vision module of the sender is trained (bool)
        :param train_vision_receiver: whether vision module of the receiver is trained (bool)
        :param sender_fixed: whether the sender is fixed completely (language learning scenario)
        """
        self.receivers = receivers
        self.senders = senders
        self.n_senders = len(senders)
        self.n_receivers = len(receivers)
        self.sender_optimizers = []
        self.receiver_optimizers = []
        for i in range(self.n_senders):
            self.sender_optimizers.append(optimizers.Adam(learning_rate=sender_lr))
        for i in range(self.n_receivers):
            self.receiver_optimizers.append(optimizers.Adam(learning_rate=receiver_lr))
        self.entropy_coeff_sender = entropy_coeff_sender
        self.entropy_coeff_receiver = entropy_coeff_receiver
        self.length_cost = length_cost
        self.vocab_size = self.senders[0].vocab_size
        self.max_message_length = self.senders[0].max_message_length
        self.classification = classification
        self.flexible_message_length = self.senders[0].flexible_message_length
        if not self.flexible_message_length:
            self.length_cost = 0
        self.train_vision_sender = train_vision_sender
        self.train_vision_receiver = train_vision_receiver
        self.sender_fixed = sender_fixed

    def train_epoch(self, data_loader):
        """ Train one epoch.

        :param data_loader: data loader for sender and receiver input (tf data loader)
        :return:    mean rewards
                    mean message length
                    mean sender loss
                    mean receiver loss
        """

        sender_loss_epoch = []
        receiver_loss_epoch = []
        rewards_epoch = []
        message_length_epoch = []

        for batch in data_loader:

            s_idx = np.random.randint(self.n_senders)
            r_idx = np.random.randint(self.n_receivers)

            with tf.GradientTape(persistent=True) as tape:

                if not self.sender_fixed:
                    tape.watch([self.senders[s_idx].trainable_variables, self.receivers[r_idx].trainable_variables])
                else:
                    tape.watch(self.receivers[r_idx].trainable_variables)
                if not self.classification:
                    sender_input, receiver_input, labels = batch
                else:
                    sender_input, receiver_input, labels, class_labels_sender, class_labels_receiver = batch

                message, sender_logits, entropy_sender, message_length, _ = self.senders[s_idx].forward(sender_input)
                selection, receiver_logits, entropy_receiver = self.receivers[r_idx].forward(message, receiver_input)

                rewards_orig = tf.reduce_sum(labels * selection, axis=1)
                std = tf.math.reduce_std(rewards_orig)
                if std > 0:
                    rewards = (rewards_orig - tf.reduce_mean(rewards_orig)) / std
                else:
                    rewards = (rewards_orig - tf.reduce_mean(rewards_orig))

                sender_policy_loss = - tf.reduce_mean(
                    sender_logits * tf.expand_dims(rewards - self.length_cost * message_length, axis=1)
                )

                receiver_policy_loss = - tf.reduce_mean(
                    receiver_logits * selection * tf.expand_dims(rewards, axis=1)
                )

                sender_loss = sender_policy_loss - self.entropy_coeff_sender * tf.reduce_mean(entropy_sender)
                receiver_loss = receiver_policy_loss - self.entropy_coeff_receiver * tf.reduce_mean(entropy_receiver)

                if self.classification:
                    if self.train_vision_sender:
                        prediction = self.senders[s_idx].predict_class(sender_input)
                        sender_classification_loss = tf.keras.losses.categorical_crossentropy(
                            class_labels_sender, prediction)
                        sender_loss = sender_loss + sender_classification_loss
                    if self.train_vision_receiver:
                        prediction = self.receivers[r_idx].predict_class(
                            np.array(tf.boolean_mask(receiver_input, labels))
                        )
                        receiver_classification_loss = tf.keras.losses.categorical_crossentropy(
                            class_labels_receiver, prediction)
                        receiver_loss = receiver_loss + receiver_classification_loss

            if not self.sender_fixed:
                sender_gradients = tape.gradient(sender_loss, self.senders[s_idx].trainable_variables)
                self.sender_optimizers[s_idx].apply_gradients(
                    zip(sender_gradients, self.senders[s_idx].trainable_variables)
                )

            receiver_gradients = tape.gradient(receiver_loss, self.receivers[r_idx].trainable_variables)
            self.receiver_optimizers[r_idx].apply_gradients(
                zip(receiver_gradients, self.receivers[r_idx].trainable_variables)
            )

            rewards_epoch.append(tf.reduce_mean(rewards_orig))
            sender_loss_epoch.append(tf.reduce_mean(sender_loss))
            receiver_loss_epoch.append(tf.reduce_mean(receiver_loss))
            message_length_epoch.append(tf.reduce_mean(message_length))

        return (np.mean(rewards_epoch), np.mean(message_length_epoch),
                np.mean(sender_loss_epoch), np.mean(receiver_loss_epoch))

    def evaluate(self, data_loader):
        """ evaluate agent's communication performance

        :param data_loader: tf data loader
        :return: mean test rewards
        """

        test_rewards = []

        for batch in data_loader:
            s_idx = np.random.randint(self.n_senders)
            r_idx = np.random.randint(self.n_receivers)

            sender_input, receiver_input, labels = batch
            message, _, _, _, _ = self.senders[s_idx].forward(sender_input, training=False)
            selection, _, _ = self.receivers[r_idx].forward(message, receiver_input, training=False)

            rewards = np.mean(selection == np.argmax(labels, axis=1))
            test_rewards.append(tf.reduce_mean(rewards))

        return np.mean(test_rewards)

    def evaluate_classification(self, data_loader):
        """ evaluate the agents' classification performance

        :param data_loader: tf data loader
        :return: mean accuracies for all agents that are trained on the classification task
        """

        if self.train_vision_sender and self.train_vision_receiver:
            all_accuracies = [[], []]
        else:
            all_accuracies = []

        for batch in data_loader:

            s_idx = np.random.randint(self.n_senders)
            r_idx = np.random.randint(self.n_receivers)

            agent_input, class_labels = batch

            if self.train_vision_sender:
                vision_output = self.senders[s_idx].vision_module(agent_input)
                predictions = self.senders[s_idx].classification_module(vision_output)
                class_labels_non_hot = tf.argmax(class_labels, axis=1)
                predictions_non_hot = tf.argmax(predictions, axis=1)
                if self.train_vision_receiver:
                    all_accuracies[0].append(np.mean(class_labels_non_hot == predictions_non_hot))
                else:
                    all_accuracies.append(np.mean(class_labels_non_hot == predictions_non_hot))

            if self.train_vision_receiver:
                vision_output = self.receivers[r_idx].vision_module(agent_input)
                predictions = self.receivers[r_idx].classification_module(vision_output)
                class_labels_non_hot = tf.argmax(class_labels, axis=1)
                predictions_non_hot = tf.argmax(predictions, axis=1)
                if self.train_vision_sender:
                    all_accuracies[1].append(np.mean(class_labels_non_hot == predictions_non_hot))
                else:
                    all_accuracies.append(np.mean(class_labels_non_hot == predictions_non_hot))

        if self.train_vision_sender and self.train_vision_receiver:
            return np.mean(all_accuracies, axis=1)
        else:
            return np.mean(all_accuracies)
