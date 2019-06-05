import tensorflow as tf

from .agent import Agent
from .sender_agent import SenderAgent

def cosine_similarity(a, b, axis=1):
    normalize_a = tf.nn.l2_normalize(a, axis=axis)
    normalize_b = tf.nn.l2_normalize(b, axis=axis)
    return tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=axis)


class ReceiverAgent(Agent):
    rnn_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(Agent.num_hidden)
    # Unique RNN cell and fc rnn output layer
    fc = tf.keras.layers.Dense(Agent.num_hidden, activation=tf.nn.tanh)
    layers = [fc]

    # Shared embedding and image fc layer
    embedding = None
    img_fc = tf.keras.layers.Dense(Agent.num_hidden, activation=tf.nn.tanh)
    shared_layers = [img_fc]

    if Agent.split_sr:
        layers += [rnn_cell]
    else:
        shared_layers += [rnn_cell]

    saver = None
    loaded = False

    def __init__(self, message, msg_len):
        # Get embedding variable and add it to shared variables if this hasn't been done yet
        if ReceiverAgent.embedding is None:
            ReceiverAgent.embedding = tf.get_variable(name="map", shape=[Agent.K, Agent.emb_size])
            ReceiverAgent.shared_layers += [ReceiverAgent.embedding]

        # Define input message
        self.message = tf.cast(message, tf.float32)
        self.msg_len = msg_len
        super().__init__()

    def all_agents_initialized(self, load_key=None):
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
        """
        # TODO: consider a better starting state for receiver
        self.s0 = ReceiverAgent.rnn_cell.zero_state(Agent.batch_size, dtype=tf.float32)
        self.batch_embedding = tf.tile(tf.expand_dims(ReceiverAgent.embedding, axis=0), [Agent.batch_size, 1, 1])
        self.msg_embeddings = tf.matmul(self.message, self.batch_embedding)

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
        self.rnn_outputs, self.final_state = tf.nn.dynamic_rnn(ReceiverAgent.rnn_cell, self.msg_embeddings,
                                                               initial_state=self.s0)
        # Get RNN features
        # TODO consider using final rnn_output instead of final_state (not sure which is better)
        # self.rnn_features = tf.keras.layers.Dense(self.num_hidden, activation=tf.nn.leaky_relu)(self.final_state)
        # self.rnn_features = tf.keras.layers.Dense(self.num_hidden, activation=tf.nn.leaky_relu)(self.rnn_features)

        self.rnn_features = ReceiverAgent.fc(self.final_state.c)

        # Get image features
        # TODO - can we do this with only matrices?
        self.candidates = []
        self.image_features = []
        self.energies = []

        # Get image features and energies for each candidate
        for c in range(Agent.D + 1):
            can = tf.placeholder(tf.float32, shape=(Agent.batch_size, 2048))
            self.candidates.append(can)
            img_feat = can / tf.maximum(tf.reduce_max(tf.abs(can), axis=1, keepdims=True), Agent.epsilon)
            img_feat = ReceiverAgent.img_fc(img_feat)

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
        
        self.img_feat_tensor = tf.stack(self.image_features, axis=1)
        # Get prediction
        self.prob_dist = tf.nn.softmax(self.energy_tensor + Agent.epsilon)
        self.prediction = tf.argmax(self.prob_dist, axis=1, output_type=tf.int32)

    def _build_losses(self):
        """
        Build loss function
        See https://arxiv.org/abs/1710.06922 Appendix A for discussion on losses
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
        """
        Build optimizer for sender and receiver. They share a loss function, but have different weights to update on
        """
        self.optimizers = {}
        self.train_ops = {}
        for i in range(2):
            v = SenderAgent.get_all_weights() if i == 0 else ReceiverAgent.get_all_weights()
            mode = "sender" if i == 0 else "receiver"

            self.optimizers[mode] = tf.train.AdamOptimizer()
            self.train_ops[mode] = tf.contrib.layers.optimize_loss(loss=self.loss,
                                                                   global_step=Agent.step,
                                                                   learning_rate=Agent.lr,
                                                                   optimizer=self.optimizers[mode],
                                                                   # some gradient clipping stabilizes training in the beginning.
                                                                   clip_gradients=Agent.gradient_clip,
                                                                   # only update receiver agent weights
                                                                   variables=v)
        ReceiverAgent.layers += list(self.optimizers["sender"].variables())
        ReceiverAgent.layers += list(self.optimizers["receiver"].variables())

    def get_train_ops(self):
        """
        Returns bots sender and receiver training ops because sender's train_op is defined using the receiver's loss
        """
        return [self.train_ops["sender"], self.train_ops["receiver"]]

    def get_output(self):
        return self.accuracy, self.loss, self.prediction

    def fill_feed_dict(self, fd, candidates, target_idx):
        """
        Fill necessary placeholders required for receiver
        :param fd: feed_dict to fill
        :param candidates: image for each candidate
        :param target_idx: target idx
        """
        for i, c in enumerate(candidates):
            fd[self.candidates[i]] = c
        if Agent.train:
            fd[self.target_indices] = target_idx
