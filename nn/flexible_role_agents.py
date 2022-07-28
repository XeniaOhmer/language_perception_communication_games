import tensorflow as tf
from tensorflow.keras import layers
from nn.agents import BaseAgent


class FlexibleRoleAgent(BaseAgent):
    """ The flexible-role agent performs sender and receiver function with the same vision and language module."""

    def __init__(self, vocab_size, max_message_length, embed_dim, hidden_dim, vision_module,
                 activation='tanh', train_vision=False, n_distractors=2, image_dim=64):
        """ Constructor. For most parameters see BaseAgent.

        :param activation: vision to hidden layer activation function (string)
        :param n_distractors: number of distractor images (int)
        :param image_dim: image size for nxn images (int)
        """

        super(FlexibleRoleAgent, self).__init__(vocab_size, max_message_length, embed_dim, hidden_dim, vision_module,
                                                flexible_message_length=False,
                                                VtoH_activation=activation,
                                                train_vision=train_vision)

        self.language_module = layers.GRUCell(hidden_dim, name='GRU_layer')
        if self.max_message_length > 1:
            self.embedding = layers.Embedding(vocab_size, embed_dim, name='embedding')

        # sender-specific
        self.hidden_to_output = layers.Dense(vocab_size, activation='linear', name='hidden_to_output')

        # receiver-specific
        self.n_distractors = n_distractors
        self.image_dim = image_dim

        self.__build()

    def __build(self, feature_dim=16):
        """ Initialize the network weights.

        :param feature_dim: dimension of object representation (vision module output) (int)
        """
        if self.vision_module is None:
            input_shape = [2, feature_dim]  # axis 0 has dimension 2 to simulate a minibatch
        else:
            input_shape = self.vision_module.input.get_shape().as_list()
            input_shape = [2] + input_shape[1:]
        test_input = tf.zeros(input_shape)
        _ = self.sender_forward(test_input, training=False)

    def sender_forward(self, inputs, training=True):

        """ Agent forward pass: generate a message based on visual input.

        :param inputs:
        :param training: whether training or testing (bool)
        :return:    sequence = message,
                    logits = logits for each symbol in message,
                    entropy = entropy of the policy,
                    message length = length of generated message,
                    h_t = final hidden state of language module
        """

        batch_size = tf.shape(inputs)[0]
        if self.vision_module is None:
            prev_hidden = self.vision_to_hidden(inputs)
        else:
            prev_hidden = self.vision_to_hidden(self.vision_module(inputs))
        cell_input = tf.zeros((batch_size, self.embed_dim))

        sequence = []
        logits = []
        entropy = []

        # generate first symbol then pass the last generated symbol as input at each time step
        for step in range(self.max_message_length):
            h_t, _ = self.language_module(cell_input, [prev_hidden])

            step_logits = tf.nn.log_softmax(self.hidden_to_output(h_t), axis=1)
            step_entropy = -tf.reduce_sum(step_logits * tf.exp(step_logits), axis=1)

            if training:
                symbol = tf.random.categorical(step_logits, 1)
            else:
                symbol = tf.expand_dims(tf.argmax(step_logits, axis=1), axis=1)

            logits.append(tf.gather_nd(step_logits, symbol, batch_dims=1))
            symbol = tf.squeeze(symbol)
            sequence.append(symbol)
            entropy.append(step_entropy)

            if self.max_message_length > 1:
                cell_input = self.embedding(symbol)

        sequence = tf.transpose(tf.cast(tf.stack(sequence), tf.float32), (1, 0))
        logits = tf.transpose(tf.stack(logits), (1, 0))
        entropy = tf.reduce_mean(tf.transpose(tf.stack(entropy), (1, 0)), axis=1)
        message_length = tf.ones_like(entropy) * self.max_message_length

        return sequence, logits, entropy, message_length, h_t

    def receiver_forward(self, message, images, training=True, feature_dim=16):
        """ Forward pass. Generate a selection based on a message and a set of images.

        :param message: input message from the sender (tensor)
        :param images: input images, target and distractors (tensor)
        :param training: whether training or testing (boolean)
        :param feature_dim: dimension of object representation, i.e. vision module output dimension (int)
        :return actions: selection or selection policy
                logits: logits for actions
                entropy: entropy of selection policy
        """

        batch_size = tf.shape(message)[0]
        prev_hidden = tf.zeros((batch_size, self.hidden_dim))

        # generate first symbol then pass the last generated symbol as input at each time step
        for step in range(self.max_message_length):
            cell_input = self.embedding(message[:, step])
            message_embeddings, _ = self.language_module(cell_input, [prev_hidden])
            prev_hidden = message_embeddings

        if self.vision_module is not None:
            image_embeddings = self.vision_to_hidden(
                self.vision_module(tf.reshape(images, (-1, self.image_dim, self.image_dim, 3)))
            )
            image_embeddings = tf.reshape(image_embeddings, (-1, self.n_distractors + 1, self.hidden_dim))
        else:
            image_embeddings = self.vision_to_hidden(
                tf.reshape(images, (-1, feature_dim))
            )
            image_embeddings = tf.reshape(image_embeddings, (-1, self.n_distractors + 1, self.hidden_dim))
        similarities = tf.reduce_sum(image_embeddings * tf.expand_dims(message_embeddings, axis=1), axis=2)
        logits = tf.nn.log_softmax(similarities, axis=1)
        entropy = - tf.reduce_sum(logits * tf.exp(logits), axis=1)
        if training:
            actions = tf.squeeze(tf.one_hot(tf.random.categorical(logits, 1), depth=self.n_distractors + 1))
        else:
            actions = tf.argmax(logits, axis=1)
        return actions, logits, entropy


class ClassificationFlexibleRoleAgent(FlexibleRoleAgent):
    """ The ClassificationFlexibleRoleAgent inherits from FlexibleRoleAgent and has additional
        methods and layers for performing an object classification task with its vision module.
    """

    def __init__(self, vocab_size, max_message_length, embed_dim, hidden_dim, vision_module,
                 classification_module, activation='tanh', train_vision=True, n_distractors=2, image_dim=64):
        """ Constructor. For most params see FlexibleRoleAgent.

        :param classification_module: classification module can be passed (tf neural network)
        The classification module is a softmax output layer that is appended to the vision module
        """

        super(ClassificationFlexibleRoleAgent, self).__init__(vocab_size,
                                                              max_message_length,
                                                              embed_dim,
                                                              hidden_dim,
                                                              vision_module,
                                                              activation=activation,
                                                              train_vision=train_vision,
                                                              image_dim=image_dim,
                                                              n_distractors=n_distractors)

        self.classification_module = classification_module
        self.__build_classification_module()

    def __build_classification_module(self):
        """ Initialize classification module weights. """
        input_shape = self.vision_module.input.get_shape().as_list()
        input_shape = [2] + input_shape[1:]
        test_input = tf.zeros(input_shape)
        _ = self.predict_class(test_input)

    def predict_class(self, image):
        """ Predict the object class based on an input image"""
        vision_output = self.vision_module(image)
        classification = self.classification_module(vision_output)
        return classification
