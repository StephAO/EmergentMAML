from Agents.agent import Agent
from data_handler import img_w, img_h
import numpy as np
import tensorflow as tf


def cosine_similarity(a, b, axis=1):
    normalize_a = tf.nn.l2_normalize(a, axis=axis)
    normalize_b = tf.nn.l2_normalize(b, axis=axis)
    return tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=axis)

class ReceiverAgent(Agent):

    def __init__(self, vocab_size, num_distractors, message, msg_len, use_images=False):
        self.query_key_size = 128
        self.message = message
        self.msg_len = msg_len
        super().__init__(vocab_size, num_distractors, use_images=use_images)

    def _build_input(self):
        """
        Build starting state and inputs to next state
        For receiver agent:
            - Starting state is the zero state.
            - Inputs the message passed from the sender
        :return: None
        """
        # TODO: consider a better starting state for receiver
        self.s0 = tf.zeros((self.batch_size, self.num_hidden), dtype=tf.float32)

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
        self.rnn_outputs, self.final_state = tf.nn.dynamic_rnn(self.gru_cell, self.message, initial_state=self.s0, time_major=True)
                                                               # sequence_length=self.msg_len, time_major=True)
        # Get RNN features
        # TODO consider using final rnn_output instead of final_state (not sure which is better)
        self.rnn_features = tf.keras.layers.Dense(self.num_hidden, activation=tf.nn.leaky_relu,
                                            kernel_initializer=tf.glorot_uniform_initializer)(self.final_state)
        self.rnn_features = tf.keras.layers.Dense(self.num_hidden, activation=tf.nn.leaky_relu,
                                            kernel_initializer=tf.glorot_uniform_initializer)(self.rnn_features)
        self.rnn_features = tf.keras.layers.Dense(self.query_key_size, activation=tf.nn.tanh,
                                            kernel_initializer=tf.glorot_uniform_initializer)(self.rnn_features)


        # TODO: consider adding noise to rnn features - is this different than just changing temperature?
        # self.rnn_features = tf.keras.layers.GaussianNoise(stddev=0.0001)(self.rnn_features)

        # Get image features
        # TODO - can we do this with only matrices?
        self.candidates = []
        self.image_features = []
        self.img_feat_1 = []# delete
        self.energies = []
        self.img_trans_1 = tf.keras.layers.Dense(self.num_hidden, activation=tf.nn.leaky_relu,
                                           kernel_initializer=tf.random_normal_initializer)
        self.img_trans_2 = tf.keras.layers.Dense(self.query_key_size, activation=tf.nn.tanh,
                                           kernel_initializer=tf.random_normal_initializer)
        for d in range(self.D + 1):
            if self.use_images:
                can = tf.placeholder(tf.float32, shape=(self.batch_size, img_h, img_w, 3))
                self.candidates.append(can)
                img_feat = Agent.pre_trained(can)
                if self.freeze_cnn:
                    img_feat = tf.stop_gradient(img_feat)
                # img_feat = tf.keras.layers.Conv2D(128, (3, 3))(img_feat)
                # img_feat = tf.keras.layers.GlobalAveragePooling2D()(img_feat)
                img_feat = img_feat / tf.maximum(tf.reduce_max(img_feat, axis=1, keepdims=True), self.epsilon)

            else: # use one-hot encoding
                idx = tf.fill([self.batch_size], d)
                img_feat = tf.one_hot(idx, self.D+1)

            self.img_feat_1.append(img_feat)

            img_feat = self.img_trans_1(img_feat)
            img_feat = self.img_trans_2(img_feat)

            # TODO: Consider adding adding noise to imgage features - is this different than just changing temperature?
            self.image_features.append(img_feat)

            # Define energies
            if self.loss_type == "pairwise":
                self.energies.append(cosine_similarity(self.rnn_features, img_feat, axis=1))
            elif self.loss_type == "MSE":
                e = tf.negative(tf.reduce_sum(tf.pow(tf.abs(tf.subtract(self.rnn_features, img_feat) + 1e-8), 0.5), axis=1))
                self.energies.append(e)
            elif self.loss_type == "invMSE":
                e = tf.reduce_sum(tf.divide(1., tf.pow(tf.subtract(self.rnn_features, img_feat), 2) + 1e-8), axis=1)
                self.energies.append(e)

        #
        self.energy_tensor = tf.stack(self.energies, axis=1)
        # self.energy_tensor = tf.divide(self.energy_tensor, tf.reduce_max(tf.abs(self.energy_tensor), axis=1, keepdims=True))
        self.img_feat_tensor = tf.stack(self.image_features, axis=1)
        # Get prediction
        self.prob_dist = tf.nn.softmax(self.energy_tensor + 1e-8)
        self.output = tf.argmax(self.prob_dist, axis=1)

    def _build_losses(self):
        """
        Build loss function
        See https://arxiv.org/abs/1710.06922 Appendix A for discussion on losses
        Currently only pairwise is working
        :return:
        """
        self.target_indices = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.ti = tf.stack([tf.range(self.batch_size), self.target_indices], axis=1)
        self.target_energy = tf.gather_nd(self.energy_tensor, self.ti)

        if self.loss_type == "pairwise":
            # Hinge loss function using Cosine similarity
            # Loss taken from https://arxiv.org/abs/1705.11192
            # Using relu to clip values below 0 since tf doesn't have a function to clip only 1 side
            # TODO the 1 part of this dominates the loss
            loss = tf.nn.relu(1 - tf.expand_dims(self.target_energy, axis=1) + self.energy_tensor)

        elif self.loss_type == "MSE":
            # Classic mse loss
            loss = float(self.D) - self.D * self.target_energy + tf.reduce_sum(self.energy_tensor, axis=1)

        elif self.loss_type == "invMSE":
            # loss taken from https://arxiv.org/abs/1710.06922 - supposed to better than others
            loss = tf.negative(tf.log(self.prob_dist + 1e-8))

        self.loss = (tf.reduce_sum(loss) / self.batch_size)

    def _build_optimizer(self):
        self.train_op = tf.contrib.layers.optimize_loss(
            loss=self.loss,
            global_step=self.epoch,
            learning_rate=self.lr,
            optimizer="Adam",
            # some gradient clipping stabilizes training in the beginning.
            clip_gradients=self.gradient_clip)

    def fill_feed_dict(self, fd, candidates, target_idx):
        fd[self.target_indices] = target_idx
        if self.use_images:
            for i, c in enumerate(candidates):
                fd[self.candidates[i]] = c

    def close(self):
        self.sess.close()

    def __del__(self):
        self.sess.close()