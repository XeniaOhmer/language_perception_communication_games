import tensorflow as tf
from tensorflow.keras import Sequential, layers, Model


class BaseAgent(Model):

    def __init__(self, vocab_size, max_message_length, embed_dim, hidden_dim, vision_module=None,
                 flexible_message_length=False, VtoH_activation='tanh', train_vision=False):
        """Constructor.

        :param vocab_size: number of symbols (int)
        :param max_message_length: maximal message length (int)
        :param embed_dim: dimension of embedding layer in language module (int)
        :param hidden_dim: dimension of hidden layer in language module (int)
        :param vision_module: vision module can be passed (tf neural network)
        :param flexible_message_length: whether message length is fixed or flexible,
        if False, max_message_length corresponds to the message length (boolean)
        :param VtoH_activation: vision to hidden mapping function activation (string)
        :param train_vision: whether the vision module should be trained or not (boolean)
        """
        super(BaseAgent, self).__init__()
        self.vocab_size = vocab_size
        self.max_message_length = max_message_length
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vision_module = vision_module
        self.flexible_message_length = flexible_message_length
        if vision_module is not None and not train_vision:
            self.vision_module.trainable = False
        self.vision_to_hidden = layers.Dense(hidden_dim, activation=VtoH_activation, name='vision_to_hidden')

    def __call__(self, inputs):
        pass

    def forward(self, *args, **kwargs):
        pass


class Sender(BaseAgent):

    def __init__(self, vocab_size, max_message_length, embed_dim, hidden_dim, vision_module,
                 flexible_message_length=False, activation='tanh', train_vision=False):
        """ Sender agent constructor. For most parameters see BaseAgent.

        :param activation: vision to hidden mapping function activation (string)
        """

        super(Sender, self).__init__(vocab_size, max_message_length, embed_dim, hidden_dim, vision_module,
                                     flexible_message_length=flexible_message_length,
                                     VtoH_activation=activation, train_vision=train_vision)
        self.language_module = layers.GRUCell(hidden_dim, name='GRU_layer')
        self.hidden_to_output = layers.Dense(vocab_size, activation='linear', name='hidden_to_output')  # must be linear
        if self.max_message_length > 1:
            self.embedding = layers.Embedding(vocab_size, embed_dim, name='embedding')
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
        _ = self.forward(test_input, training=False)

    def __unroll(self, cell_input, prev_hidden, training=False):
        """ unroll the RNN cell

        :param cell_input: initial input
        :param prev_hidden: hidden state initialization
        :param training: whether agent is being trained
        :return:
            logits - logits for the symbols at each time step
            entropy - entropy for the symbols at each time step
            sequence - symbols over time steps
            symbol_mask - mask to remove symbols after eos (if message length flexible)
            h_t - final hidden state
        """

        # generate first symbol then pass the last generated symbol as input at each time step

        sequence = []
        logits = []
        entropy = []
        symbol_mask = []

        for step in range(self.max_message_length):

            h_t, _ = self.language_module(cell_input, [prev_hidden])

            step_logits = tf.nn.log_softmax(self.hidden_to_output(h_t), axis=1)
            step_entropy = -tf.reduce_sum(step_logits * tf.exp(step_logits), axis=1)

            if training:
                symbol = tf.random.categorical(step_logits, 1)
            else:
                symbol = tf.expand_dims(tf.argmax(step_logits, axis=1), axis=1)

            symbol_mask.append(tf.squeeze(tf.cast(symbol == 0, dtype=tf.int32)))

            logits.append(tf.gather_nd(step_logits, symbol, batch_dims=1))
            symbol = tf.squeeze(symbol)
            sequence.append(symbol)
            entropy.append(step_entropy)

            cell_input = self.embedding(symbol)
            prev_hidden = h_t

        return logits, entropy, sequence, symbol_mask, h_t

    def forward(self, inputs, training=True):
        """ Agent forward pass: generate a message based on visual input.

        :param inputs:
        :param training: whether training or testing (bool)
        :return:    sequence = message,
                    logits = logits for each symbol in message,
                    entropy = entropy of the policy,
                    message length = length of generated mesage,
                    h_t = final hidden state of language module
        """

        batch_size = tf.shape(inputs)[0]
        if self.vision_module is None:
            init_hidden = self.vision_to_hidden(inputs)
        else:
            init_hidden = self.vision_to_hidden(self.vision_module(inputs))
        cell_input = tf.zeros((batch_size, self.embed_dim))

        logits, entropy, sequence, symbol_mask, h_t = self.__unroll(cell_input, init_hidden)

        if self.flexible_message_length:
            cumsum = tf.cast(tf.cumsum(tf.stack(symbol_mask, axis=0), axis=0), tf.float32)
            # calculate mask ignoring zeros to determine actual message length
            mask = tf.cast(cumsum == 0, tf.float32)
            message_length = tf.reduce_sum(mask, axis=0)
            # calculate mask including final zero to calculate relevant policies and entropies
            eos_padding = tf.zeros((1, cumsum.shape[1]))
            cumsum = tf.concat([eos_padding, cumsum[:-1, :]], axis=0)
            mask = tf.cast(cumsum == 0, tf.float32)
            message_length_with_zeros = tf.reduce_sum(mask, axis=0)
            sequence = tf.transpose(tf.cast(tf.stack(sequence), tf.float32) * mask, (1, 0))
            logits = tf.transpose(tf.stack(logits) * mask, (1, 0))
            entropy = tf.reduce_sum(tf.transpose(tf.stack(entropy) * mask, (1, 0)), axis=1) / message_length_with_zeros
        else:
            sequence = tf.transpose(tf.cast(tf.stack(sequence), tf.float32), (1, 0))
            logits = tf.transpose(tf.stack(logits), (1, 0))
            entropy = tf.reduce_mean(tf.transpose(tf.stack(entropy), (1, 0)), axis=1)
            message_length = tf.ones_like(entropy) * self.max_message_length

        return sequence, logits, entropy, message_length, h_t


class ClassificationSender(Sender):
    """ The ClassificationSender inherits from Sender and has additional
        methods and layers for performing an object classification task with its vision module.
    """

    def __init__(self, vocab_size, max_message_length, embed_dim, hidden_dim, vision_module,
                 classification_module, flexible_message_length=False, activation='tanh', train_vision=True):
        """ Constructor. For most parameters see Sender.

        :param classification_module: classification module can be passed (tf neural network)
        The classification module is a softmax output layer that is appended to the vision module
        """

        super(ClassificationSender, self).__init__(
            vocab_size, max_message_length, embed_dim, hidden_dim, vision_module,
            flexible_message_length=flexible_message_length, activation=activation, train_vision=train_vision)

        self.classification_module = classification_module
        self.__build_classification_module()

    def __build_classification_module(self):
        """ Initialize classification module weights."""
        input_shape = self.vision_module.input.get_shape().as_list()
        input_shape = [2] + input_shape[1:]
        test_input = tf.zeros(input_shape)
        _ = self.predict_class(test_input)

    def predict_class(self, image):
        """ Predict the object class based on an input image"""
        vision_output = self.vision_module(image)
        classification = self.classification_module(vision_output)
        return classification


class Receiver(BaseAgent):

    def __init__(
            self, vocab_size, max_message_length, embed_dim, hidden_dim, vision_module, flexible_message_length=False,
            activation='tanh', n_distractors=2, image_dim=64, train_vision=False
    ):
        """ Constructor. For most parameters see BaseAgent.

        :param activation: vision to hidden layer activation function (string)
        :param n_distractors: number of distractor images (int)
        :param image_dim: image size for nxn images (int)
        """
        super(Receiver, self).__init__(vocab_size, max_message_length, embed_dim, hidden_dim, vision_module,
                                       flexible_message_length=flexible_message_length,
                                       VtoH_activation=activation, train_vision=train_vision)
        self.n_distractors = n_distractors
        self.image_dim = image_dim
        # if message length fixed, 0 counts as a standard symbol and no masking is applied
        self.language_module = Sequential([layers.Embedding(vocab_size, embed_dim,
                                                            mask_zero=self.flexible_message_length,
                                                            name='embedding'),
                                           layers.GRU(hidden_dim, name='GRU_layer')])
        self.__build()

    def __build(self, feature_dim=16):
        """ Initialize the network weights

        :param feature_dim: dimension of visual representations
        """
        if self.vision_module is None:
            input_shape = [2, self.n_distractors + 1, feature_dim]
        else:
            input_shape = self.vision_module.input.get_shape().as_list()
            input_shape = [2, self.n_distractors + 1] + input_shape[1:]
        test_input = tf.zeros(input_shape)
        test_messages = tf.zeros((2, self.max_message_length))
        _ = self.forward(test_messages, test_input, training=False)

    def forward(self, message, images, training=True, feature_dim=16):
        """ Forward pass. Generate a selection based on a message and a set of images.

        :param message: input message from the sender (tensor)
        :param images: input images, target and distractors (tensor)
        :param training: whether training or testing (boolean)
        :param feature_dim: dimension of object representation, i.e. vision module output dimension (int)
        :return actions: selection or selection policy
                logits: logits for actions
                entropy: entropy of selection policy
        """
        message_embeddings = self.language_module(message)
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


class ClassificationReceiver(Receiver):
    """ The ClassificationReceiver inherits from Receiver and has additional
        methods and layers for performing an object classification task with its vision module."""

    def __init__(self, vocab_size, max_message_length, embed_dim, hidden_dim, vision_module, classification_module,
                 flexible_message_length=False, activation='tanh', n_distractors=2, image_dim=64, train_vision=True):
        """ Constructor. For most params see Receiver.

        :param classification_module: classification module can be passed (tf neural network)
        The classification module is a softmax output layer that is appended to the vision module
        """
        super(ClassificationReceiver, self).__init__(
            vocab_size, max_message_length, embed_dim, hidden_dim, vision_module, activation=activation,
            flexible_message_length=flexible_message_length, train_vision=train_vision, image_dim=image_dim,
            n_distractors=n_distractors)

        self.classification_module = classification_module
        self.__build_classification_module()

    def __build_classification_module(self):
        """ Intialize classification module weights"""
        input_shape = self.vision_module.input.get_shape().as_list()
        input_shape = [2] + input_shape[1:]
        test_input = tf.zeros(input_shape)
        _ = self.predict_class(test_input)

    def predict_class(self, image):
        """ Predict the object class based on an input image"""
        vision_output = self.vision_module(image)
        classification = self.classification_module(vision_output)
        return classification


class FlexibleRoleAgent(Sender):
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
                                                activation=activation,
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

        return self.forward(inputs, training=training)

    def __receiver_unroll(self, prev_hidden, message):
        """ Unroll receiver RNN cell when processing message

        :param prev_hidden: initial hidden state
        :param message: message
        :return: message_embeddings : final hidden state
        """

        for step in range(self.max_message_length):
            cell_input = self.embedding(message[:, step])
            message_embeddings, _ = self.language_module(cell_input, [prev_hidden])
            prev_hidden = message_embeddings

        return message_embeddings

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
        hidden_init = tf.zeros((batch_size, self.hidden_dim))

        message_embeddings = self.__receiver_unroll(hidden_init, message)

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
