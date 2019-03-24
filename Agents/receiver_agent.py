import tensorflow as tf

from .agent import Agent
from .sender_agent import SenderAgent
from utils.data_handler import img_w, img_h


def cosine_similarity(a, b, axis=1):
    normalize_a = tf.nn.l2_normalize(a, axis=axis)
    normalize_b = tf.nn.l2_normalize(b, axis=axis)
    return tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=axis)


class ReceiverAgent(Agent):
    # Used to map RNN output to RNN features
    fc = tf.keras.layers.Dense(Agent.num_hidden, activation=tf.nn.tanh,
                               kernel_initializer=tf.glorot_uniform_initializer)
    layers = [fc]

    # Shared image fc layer
    img_fc = tf.keras.layers.Dense(Agent.num_hidden, activation=tf.nn.tanh,
                                   kernel_initializer=tf.glorot_uniform_initializer)
    # img_fc = tf.make_template("img_fc", img_fc)
    # img_fc.name = "shared_fc"
    # Shared RNN cell
    rnn_cell = tf.nn.rnn_cell.LSTMCell(Agent.num_hidden, initializer=tf.glorot_uniform_initializer)

    # list to store MAML layers
    shared_layers = [img_fc, rnn_cell]

    saver = None

    def __init__(self, message, msg_len, load_key=None, *args, **kwargs):
        # with tf.variable_scope("receiver"):
        self.message = message
        self.msg_len = msg_len
        super().__init__(*args, **kwargs)
        # Create saver
        ReceiverAgent.saver = ReceiverAgent.saver or tf.train.Saver(var_list=ReceiverAgent.get_all_weights())
        if load_key is not None:
            ReceiverAgent.load_model(load_key)

    def _build_input(self):
        """
        Build starting state and inputs to next state
        For receiver agent:
            - Starting state is the zero state.
            - Inputs the message passed from the sender
        :return: None
        """
        # TODO: consider a better starting state for receiver
        self.s0 = ReceiverAgent.rnn_cell.zero_state(Agent.batch_size, dtype=tf.float32)

    def _build_output(self):
        """
        Build agent output from the output of the RNN
        Take output of RNN, h_t, pass it through a mlp, g(h_t).
        Pass each image, i_k, through a pre-trained model (+ potentially more), f(i_k)
        Define energies by comparing the RNN features and image features using 1 of three methods:
            - pairwise (cosine similarity)
            - MSE (Squared euclidean distance)
            - invMSE (see https://arxiv.org/abs/1710.06922)
        Get predicted image by finding image with the highest energy
        :return:
        """
        self.rnn_outputs, self.final_state = tf.nn.dynamic_rnn(ReceiverAgent.rnn_cell, self.message, initial_state=self.s0, time_major=True)
                                                               # sequence_length=self.msg_len, time_major=True)
        # Get RNN features
        # TODO consider using final rnn_output instead of final_state (not sure which is better)
        # self.rnn_features = tf.keras.layers.Dense(self.num_hidden, activation=tf.nn.leaky_relu,
        #                                     kernel_initializer=tf.glorot_uniform_initializer)(self.final_state)
        # self.rnn_features = tf.keras.layers.Dense(self.num_hidden, activation=tf.nn.leaky_relu,
        #                                     kernel_initializer=tf.glorot_uniform_initializer)(self.rnn_features)

        self.rnn_features = ReceiverAgent.fc(self.final_state.c)

        # TODO: consider adding noise to rnn features - is this different than just changing temperature?
        # self.rnn_features = tf.keras.layers.GaussianNoise(stddev=0.0001)(self.rnn_features)

        # Get image features
        # TODO - can we do this with only matrices?
        self.candidates = []
        self.image_features = []
        self.energies = []

        for d in range(Agent.D + 1):
            if self.use_images:
                can = tf.placeholder(tf.float32, shape=(Agent.batch_size, img_h, img_w, 3))
                self.candidates.append(can)
                img_feat = Agent.pre_trained(can)
                if self.freeze_cnn:
                    img_feat = tf.stop_gradient(img_feat)
                img_feat = img_feat / tf.maximum(tf.reduce_max(img_feat, axis=1, keepdims=True), Agent.epsilon)

            else:  # use one-hot encoding
                idx = tf.fill([self.batch_size], d)
                img_feat = tf.one_hot(idx, Agent.D+1)

            img_feat = ReceiverAgent.img_fc(img_feat)

            # TODO: Consider adding adding noise to imgage features - is this different than just changing temperature?
            self.image_features.append(img_feat)

            # Define energies
            if self.loss_type == "pairwise":
                # Cosine similarity
                # Loss taken from https://arxiv.org/abs/1705.11192
                self.energies.append(cosine_similarity(self.rnn_features, img_feat, axis=1))
            elif self.loss_type == "MSE":
                # Classic mse loss
                e = tf.negative(tf.reduce_sum(tf.squared_difference(self.rnn_features, img_feat), axis=1))
                self.energies.append(e)
            elif self.loss_type == "invMSE":
                # loss taken from https://arxiv.org/abs/1710.06922 - supposed to better than others
                e = tf.divide(1, tf.maximum(
                                    tf.reduce_sum(tf.squared_difference(self.rnn_features, img_feat), axis=1),
                                    Agent.epsilon))
                self.energies.append(e)

        self.energy_tensor = tf.stack(self.energies, axis=1)

        if self.loss_type == "MSE":
            self.energy_tensor = self.energy_tensor / tf.minimum(
                tf.reduce_max(self.energy_tensor, axis=1, keepdims=True), -Agent.epsilon)

        elif self.loss_type == "invMSE":
            self.energy_tensor = self.energy_tensor / tf.maximum(
                tf.reduce_max(self.energy_tensor, axis=1, keepdims=True), Agent.epsilon)
        
        # self.energy_tensor = tf.divide(self.energy_tensor, tf.reduce_max(tf.abs(self.energy_tensor), axis=1, keepdims=True))
        self.img_feat_tensor = tf.stack(self.image_features, axis=1)
        # Get prediction
        self.prob_dist = tf.nn.softmax(self.energy_tensor + Agent.epsilon)
        self.prediction = tf.argmax(self.prob_dist, axis=1, output_type=tf.int32)

    def _build_losses(self):
        """
        Build loss function
        See https://arxiv.org/abs/1710.06922 Appendix A for discussion on losses
        :return:
        """
        self.target_indices = tf.placeholder(tf.int32, shape=[Agent.batch_size])
        self.ti = tf.stack([tf.range(Agent.batch_size), self.target_indices], axis=1)
        self.target_energy = tf.gather_nd(self.energy_tensor, self.ti)

        # TODO try loss for each timestep to help with long timesequences
        # Hinge loss function using
        # Using relu to clip values below 0 since tf doesn't have a function to clip only 1 side
        loss = tf.nn.relu(1 - tf.expand_dims(self.target_energy, axis=1) + self.energy_tensor)

        self.loss = (tf.reduce_sum(loss) / Agent.batch_size)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.target_indices), tf.float32))

    def _build_optimizer(self):
        self.sender_train_op, self.receiver_train_op = [
            tf.contrib.layers.optimize_loss(
                loss=self.loss,
                global_step=Agent.step,
                learning_rate=Agent.lr,
                optimizer="Adam",
                # some gradient clipping stabilizes training in the beginning.
                clip_gradients=Agent.gradient_clip,
                # only update receiver agent weights
                variables=v)
            for v in [SenderAgent.get_all_weights(), ReceiverAgent.get_all_weights()]
        ]

    def get_train_ops(self):
        return [self.sender_train_op, self.receiver_train_op]

    def get_output(self):
        return self.accuracy, self.loss

    def fill_feed_dict(self, fd, candidates, target_idx):
        fd[self.target_indices] = target_idx
        if self.use_images:
            for i, c in enumerate(candidates):
                fd[self.candidates[i]] = c

    def close(self):
        Agent.sess.close()

    def __del__(self):
        Agent.sess.close()