import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers


class Trainer:

    def __init__(self, sender, receiver, sender_lr=0.001, receiver_lr=0.001, length_cost=0.0,
                 entropy_coeff_sender=0.0, entropy_coeff_receiver=0.0, classification=False,
                 train_vision_sender=False, train_vision_receiver=False, sender_fixed=False):
        """ Constructor.

        :param sender: sender agent
        :param receiver: receiver agent
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
        self.receiver = receiver
        self.sender = sender
        self.sender_optimizer = optimizers.Adam(learning_rate=sender_lr)
        self.receiver_optimizer = optimizers.Adam(learning_rate=receiver_lr)
        self.entropy_coeff_sender = entropy_coeff_sender
        self.entropy_coeff_receiver = entropy_coeff_receiver
        self.length_cost = length_cost
        self.vocab_size = self.sender.vocab_size
        self.max_message_length = self.sender.max_message_length
        self.classification = classification
        self.flexible_message_length = self.sender.flexible_message_length
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

            with tf.GradientTape(persistent=True) as tape:

                if not self.sender_fixed:
                    tape.watch([self.sender.trainable_variables, self.receiver.trainable_variables])
                else:
                    tape.watch(self.receiver.trainable_variables)
                if not self.classification:
                    sender_input, receiver_input, labels = batch
                else:
                    sender_input, receiver_input, labels, class_labels_sender, class_labels_receiver = batch

                message, sender_logits, entropy_sender, message_length, _ = self.sender.forward(sender_input)
                selection, receiver_logits, entropy_receiver = self.receiver.forward(message, receiver_input)

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
                        prediction = self.sender.predict_class(sender_input)
                        sender_classification_loss = tf.keras.losses.categorical_crossentropy(
                            class_labels_sender, prediction)
                        sender_loss = sender_loss + sender_classification_loss
                    if self.train_vision_receiver:
                        prediction = self.receiver.predict_class(
                            np.array([receiver_input[i, np.argmax(labels[i]), :, :, :] for i in range(len(labels))])
                        )
                        receiver_classification_loss = tf.keras.losses.categorical_crossentropy(
                            class_labels_receiver, prediction)
                        receiver_loss = receiver_loss + receiver_classification_loss

            if not self.sender_fixed:
                sender_gradients = tape.gradient(sender_loss, self.sender.trainable_variables)
                self.sender_optimizer.apply_gradients(zip(sender_gradients, self.sender.trainable_variables))

            receiver_gradients = tape.gradient(receiver_loss, self.receiver.trainable_variables)
            self.receiver_optimizer.apply_gradients(zip(receiver_gradients, self.receiver.trainable_variables))

            rewards_epoch.append(tf.reduce_mean(rewards_orig))
            sender_loss_epoch.append(tf.reduce_mean(sender_loss))
            receiver_loss_epoch.append(tf.reduce_mean(receiver_loss))
            message_length_epoch.append(tf.reduce_mean(message_length))

        return (np.mean(rewards_epoch), np.mean(message_length_epoch),
                np.mean(sender_loss_epoch), np.mean(receiver_loss_epoch))

    def evaluate(self, data_loader):
        """ evaluate agent's communication performance

        :param data_loader: tf data loader
        :return: mean validation rewards
        """

        val_rewards = []

        for batch in data_loader:
            sender_input, receiver_input, labels = batch
            message, _, _, _, _ = self.sender.forward(sender_input, training=False)
            selection, _, _ = self.receiver.forward(message, receiver_input, training=False)

            rewards = np.mean(selection == np.argmax(labels, axis=1))
            val_rewards.append(tf.reduce_mean(rewards))

        return np.mean(val_rewards)

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

            agent_input, class_labels = batch

            if self.train_vision_sender:
                vision_output = self.sender.vision_module(agent_input)
                predictions = self.sender.classification_module(vision_output)
                class_labels_non_hot = tf.argmax(class_labels, axis=1)
                predictions_non_hot = tf.argmax(predictions, axis=1)
                if self.train_vision_receiver:
                    all_accuracies[0].append(np.mean(class_labels_non_hot == predictions_non_hot))
                else:
                    all_accuracies.append(np.mean(class_labels_non_hot == predictions_non_hot))

            if self.train_vision_receiver:
                vision_output = self.receiver.vision_module(agent_input)
                predictions = self.receiver.classification_module(vision_output)
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
