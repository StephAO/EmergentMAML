from Agents.agent import Agent
from data_handler import img_w, img_h
import numpy as np
import tensorflow as tf


def cosine_similarity(a, b, axis=1):
    normalize_a = tf.nn.l2_normalize(a, axis=axis)
    normalize_b = tf.nn.l2_normalize(b, axis=axis)
    return tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=axis)

class ReceiverAgent(Agent):

    def __init__(self, vocab_size, num_distractors, message, hum_msg, msg_len, use_images=False):
        self.query_key_size = 128
        self.message = message
        # self.message = tf.placeholder(tf.float32, (1,16,2))
        self.msg_len = msg_len
        self.human_msg = hum_msg
        super().__init__(vocab_size, num_distractors, use_images=use_images)

    def _build_input(self):
        """
        Define starting state and inputs to next state
        For receiver agent:
            - starting state is the zero state.
            - First input is a start of sentence token (sos), followed by the output of the previous timestep
        :return: None
        """
        # TODO: define a better starting state for receiver
        self.s0 = tf.zeros((self.batch_size, self.num_hidden), dtype=tf.float32)

    def _build_output(self):
        """
        Take output of RNN, h_t, pass it through a one layer perceptron, g(h_t).
        Pass each image, i_k, through a pre-trained model, then a one layer perceptron, f(i_k)
        :return: chosen index of chosen image
        """
        self.rnn_outputs, self.final_state = tf.nn.dynamic_rnn(self.gru_cell, self.message, initial_state=self.s0, time_major=True)
                                                               # sequence_length=self.msg_len, time_major=True)
        # Get RNN features
        # TODO consider using final rnn_output instead of final_state (not sure which is better)
        self.rnn_features = tf.layers.dense(self.final_state, self.num_hidden, activation=tf.nn.leaky_relu,
                                            kernel_initializer=tf.glorot_uniform_initializer)
        self.rnn_features = tf.layers.dense(self.rnn_features, self.num_hidden, activation=tf.nn.leaky_relu,
                                            kernel_initializer=tf.glorot_uniform_initializer)
        self.rnn_features = tf.layers.dense(self.rnn_features, self.query_key_size, activation=tf.nn.tanh,
                                            kernel_initializer=tf.glorot_uniform_initializer)


        # Add noise?
        # self.rnn_features = tf.keras.layers.GaussianNoise(stddev=0.0001)(self.rnn_features)

        # Get image features
        # TODO - can we do this with only matrices?
        self.candidates = []
        self.image_features = []
        self.img_feat_1 = []# delete
        self.energies = []
        self.img_trans_1 = tf.layers.Dense(self.num_hidden, activation=tf.nn.leaky_relu,
                                         kernel_initializer=tf.random_normal_initializer)
        self.img_trans_2 = tf.layers.Dense(self.query_key_size, activation=tf.nn.tanh,
                                         kernel_initializer=tf.random_normal_initializer)
        self.freeze_cnn = True
        for d in range(self.D + 1):
            if self.use_images:
                can = tf.placeholder(tf.float32, shape=(self.batch_size, img_h, img_w, 3))
                self.candidates.append(can)
                img_feat = Agent.pre_trained(can)
                if self.freeze_cnn:
                    img_feat = tf.stop_gradient(img_feat)
                img_feat = img_feat / tf.reduce_max(img_feat, axis=1, keepdims=True)
                # img_feat_1 = tf.layers.max_pooling2d(can, (16, 16), (16, 16))
                # img_feat_1 = tf.layers.flatten(img_feat_1)

            else: # use one-hot encoding
                idx = tf.fill([self.batch_size], d)
                img_feat = tf.one_hot(idx, self.D+1)

            self.img_feat_1.append(img_feat)

            img_feat = self.img_trans_1(img_feat)
            img_feat = self.img_trans_2(img_feat)

            # Try adding noise to img_feat and rnn_feat
            self.image_features.append(img_feat)

            if self.loss_type == "pairwise":
                self.energies.append(cosine_similarity(self.rnn_features, img_feat, axis=1))
            elif self.loss_type == "MSE":
                e = tf.negative(tf.reduce_sum(tf.pow(tf.abs(tf.subtract(self.rnn_features, img_feat) + 1e-8), 0.5), axis=1))
                self.energies.append(e)
            elif self.loss_type == "invMSE":
                e = tf.reduce_sum(tf.divide(1., tf.pow(tf.subtract(self.rnn_features, img_feat), 2) + 1e-8), axis=1)
                self.energies.append(e)


        self.energy_tensor = tf.stack(self.energies, axis=1)
        # self.energy_tensor = tf.divide(self.energy_tensor, tf.reduce_max(tf.abs(self.energy_tensor), axis=1, keepdims=True))
        self.img_feat_tensor = tf.stack(self.image_features, axis=1)

        self.prob_dist = tf.nn.softmax(self.energy_tensor + 1e-8)

        self.output = tf.argmax(self.prob_dist, axis=1)

    def _build_losses(self):
        # See https://arxiv.org/abs/1710.06922 Appendix A for discussion on losses
        # Currently only pairwise is having any success
        self.target_indices = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.ti = tf.stack([tf.range(self.batch_size), self.target_indices], axis=1)
        self.target_energy = tf.gather_nd(self.energy_tensor, self.ti)

        if self.loss_type == "pairwise":
            # Dot product loss
            # Loss taken from https://arxiv.org/abs/1705.11192
            # Using relu to clip values below 0 since tf doesn't have a function to clip only 1 side
            # TODO the 1 part of this dominates the loss
            loss = tf.nn.relu(1 - tf.expand_dims(self.target_energy, axis=1) + self.energy_tensor)

        elif self.loss_type == "MSE":
            loss = float(self.D) - self.D * self.target_energy + tf.reduce_sum(self.energy_tensor, axis=1)

        elif self.loss_type == "invMSE":
            loss = tf.negative(tf.log(self.prob_dist + 1e-8))

        # maximize cosine distance between image features
        # a = tf.squeeze(tf.slice(self.img_feat_tensor, [0,0,0], [16, 1, 128]))
        # b = tf.squeeze(tf.slice(self.img_feat_tensor, [0,1,0], [16, 1, 128]))
        # self.diff_loss = tf.reduce_sum(cosine_similarity(a, b, axis=1)) / self.batch_size

        self.loss = (tf.reduce_sum(loss) / self.batch_size) # + self.diff_loss

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

        # a = np.zeros((1,16,2))
        # a[0, np.arange(16), target_idx] = 1
        # fd[self.message] = a

    def close(self):
        self.sess.close()

    def __del__(self):
        self.sess.close()