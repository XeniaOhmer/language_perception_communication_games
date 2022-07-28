import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers


class FlexibleRoleTrainer:

    def __init__(self, agents, lr=0.001, entropy_coeff_sender=0.0, entropy_coeff_receiver=0.0,
                 classification=False, train_vision=False):
        """ Constructor.

        :param agents: list of the two agents
        :param lr: learning rate for each agent
        :param entropy_coeff_sender: weight of entropy regularization for sender (float)
        :param entropy_coeff_receiver: weight of entropy regularization for receiver (float)
        :param classification: whether (at least one of) the agents are trained on the classification task (bool)
        :param train_vision: whether vision module of the agents is trained (bool)
        """
        self.agents = agents
        self.optimizers = [optimizers.Adam(learning_rate=lr), optimizers.Adam(learning_rate=lr)]
        self.entropy_coeff_sender = entropy_coeff_sender
        self.entropy_coeff_receiver = entropy_coeff_receiver
        self.vocab_size = self.agents[0].vocab_size
        self.max_message_length = self.agents[0].max_message_length
        self.classification = classification
        self.train_vision = train_vision

    def train_epoch(self, data_loader):
        """ Train one epoch.

        :param data_loader: data loader for sender and receiver input (tf data loader)
        :return:    mean rewards
                    mean message length
                    mean sender loss
                    mean receiver loss
        """

        sender_loss_epoch = [[], []]
        receiver_loss_epoch = [[], []]
        rewards_epoch = []
        message_length_epoch = []

        agent_index = [0, 1]

        for batch in data_loader:

            np.random.shuffle(agent_index)
            s_idx = agent_index[0]
            r_idx = agent_index[1]

            self.agents[r_idx].hidden_to_output.trainable = False
            self.agents[s_idx].hidden_to_output.trainable = True

            with tf.GradientTape(persistent=True) as tape:

                tape.watch([self.agents[0].trainable_variables, self.agents[1].trainable_variables])

                if self.train_vision:
                    if not self.classification:
                        sender_input, receiver_input, labels = batch
                    else:
                        sender_input, receiver_input, labels, class_labels_agent1, class_labels_agent2 = batch
                else:
                    a1_sender_in, a2_receiver_in, labels_a1_a2, a2_sender_in, a1_receiver_in, labels_a2_a1 = batch
                    sender_input = [a1_sender_in, a2_sender_in][s_idx]
                    receiver_input = [a1_receiver_in, a2_receiver_in][r_idx]
                    labels = [labels_a1_a2, labels_a2_a1][s_idx]

                message, sender_logits, entropy_sender, message_length, _ = self.agents[s_idx].sender_forward(sender_input)
                selection, receiver_logits, entropy_receiver = self.agents[r_idx].receiver_forward(message, receiver_input)

                rewards_orig = tf.reduce_sum(labels * selection, axis=1)
                std = tf.math.reduce_std(rewards_orig)
                if std > 0:
                    rewards = (rewards_orig - tf.reduce_mean(rewards_orig)) / std
                else:
                    rewards = (rewards_orig - tf.reduce_mean(rewards_orig))

                sender_policy_loss = - tf.reduce_mean(
                    sender_logits * tf.expand_dims(rewards, axis=1)
                )

                receiver_policy_loss = - tf.reduce_mean(
                    receiver_logits * selection * tf.expand_dims(rewards, axis=1)
                )

                sender_loss = sender_policy_loss - self.entropy_coeff_sender * tf.reduce_mean(entropy_sender)
                receiver_loss = receiver_policy_loss - self.entropy_coeff_receiver * tf.reduce_mean(entropy_receiver)

                if self.classification:
                    class_labels_sender = [class_labels_agent1, class_labels_agent2][s_idx]
                    class_labels_receiver = [class_labels_agent1, class_labels_agent2][r_idx]

                    prediction = self.agents[s_idx].predict_class(sender_input)
                    sender_classification_loss = tf.keras.losses.categorical_crossentropy(
                        class_labels_sender, prediction)
                    sender_loss = sender_loss + sender_classification_loss

                    prediction = self.agents[r_idx].predict_class(
                        np.array(tf.boolean_mask(receiver_input, labels))
                    )
                    receiver_classification_loss = tf.keras.losses.categorical_crossentropy(
                        class_labels_receiver, prediction)
                    receiver_loss = receiver_loss + receiver_classification_loss

            sender_gradients = tape.gradient(sender_loss, self.agents[s_idx].trainable_variables)
            self.optimizers[s_idx].apply_gradients(zip(sender_gradients, self.agents[s_idx].trainable_variables))

            receiver_gradients = tape.gradient(receiver_loss, self.agents[r_idx].trainable_variables)
            self.optimizers[r_idx].apply_gradients(zip(receiver_gradients, self.agents[r_idx].trainable_variables))

            rewards_epoch.append(tf.reduce_mean(rewards_orig))
            sender_loss_epoch[s_idx].append(tf.reduce_mean(sender_loss))
            receiver_loss_epoch[r_idx].append(tf.reduce_mean(receiver_loss))
            message_length_epoch.append(tf.reduce_mean(message_length))

        return (np.mean(rewards_epoch), np.mean(message_length_epoch),
                np.array([np.mean(sender_loss_epoch[0]), np.mean(sender_loss_epoch[1])]),
                np.array([np.mean(receiver_loss_epoch[0]), np.mean(receiver_loss_epoch[1])]))

    def evaluate(self, data_loader):
        """ evaluate agent's communication performance

        :param data_loader: tf data loader
        :return: mean validation rewards
        """

        val_rewards = []
        agent_index = [0, 1]

        for batch in data_loader:

            np.random.shuffle(agent_index)
            s_idx = agent_index[0]
            r_idx = agent_index[1]

            if self.train_vision:
                sender_input, receiver_input, labels = batch
            else:
                a1_sender_in, a2_receiver_in, labels_a1_a2, a2_sender_in, a1_receiver_in, labels_a2_a1 = batch
                sender_input = [a1_sender_in, a2_sender_in][s_idx]
                receiver_input = [a1_receiver_in, a2_receiver_in][r_idx]
                labels = [labels_a1_a2, labels_a2_a1][s_idx]

            message, _, _, _, _ = self.agents[s_idx].sender_forward(sender_input, training=False)
            selection, _, _ = self.agents[r_idx].receiver_forward(message, receiver_input, training=False)

            rewards = np.mean(selection == np.argmax(labels, axis=1))
            val_rewards.append(tf.reduce_mean(rewards))

        return np.mean(val_rewards)

    def evaluate_classification(self, data_loader):
        """ evaluate the agents' classification performance

        :param data_loader: tf data loader
        :return: mean accuracies for all agents that are trained on the classification task
        """

        all_accuracies = [[], []]

        for batch in data_loader:

            agent_input, class_labels = batch

            for i in range(2):

                vision_output = self.agents[i].vision_module(agent_input)
                predictions = self.agents[i].classification_module(vision_output)
                class_labels_non_hot = tf.argmax(class_labels, axis=1)
                predictions_non_hot = tf.argmax(predictions, axis=1)
                all_accuracies[i].append(np.mean(class_labels_non_hot == predictions_non_hot))

        return np.mean(all_accuracies, axis=1)
