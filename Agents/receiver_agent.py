from Agents.agent import Agent
from data_handler import img_w, img_h
import tensorflow as tf

class ReceiverAgent(Agent):

    def __init__(self, vocab_size, num_distractors, message, hum_msg,msg_len):
        self.query_key_size = 64
        self.message = message
        self.msg_len = msg_len
        self.human_msg = hum_msg
        super().__init__(vocab_size, num_distractors)

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
        Use -f(i_k)^T g(h_t) as an energy function
        :return: chosen index of chosen image
        """
        self.rnn_outputs, self.final_state = tf.nn.dynamic_rnn(self.gru_cell, self.message, initial_state=self.s0,
                                                               sequence_length=self.msg_len, time_major=True)
        # Get RNN features
        # TODO consider using final rnn_output instead of final_state (not sure which is better)
        self.rnn_features = tf.layers.dense(self.final_state, self.query_key_size, activation=tf.nn.relu,
                                            kernel_initializer=tf.glorot_uniform_initializer)

        # Get image features
        # TODO - can we do this with only matrices?
        self.candidates = []
        self.image_features = []
        self.energies = []
        self.img_trans = tf.layers.Dense(self.query_key_size, activation=tf.nn.relu,
                                         kernel_initializer=tf.glorot_uniform_initializer)

        for d in range(self.D + 1):
            can = tf.placeholder(tf.float32, shape=(self.batch_size, img_h, img_w, 3))
            self.candidates.append(can)
            img_feat = Agent.pre_trained(can)
            img_feat = self.img_trans(img_feat)
            self.image_features.append(img_feat)

            self.energies.append(tf.reduce_sum(tf.multiply(img_feat, self.rnn_features), axis=1))

        self.energy_tensor = tf.stack(self.energies, axis=1)

        self.prob_dist = tf.nn.softmax(self.energy_tensor)

        self.output = tf.argmax(self.prob_dist, axis=1)

    def _build_losses(self):
        self.target_indices = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.ti = tf.stack([tf.range(self.batch_size), self.target_indices], axis=1)
        self.target_energy = tf.gather_nd(self.energy_tensor, self.ti)
        # self.loss_components = []
        # for d in range(self.D):
        #     # if d == self.target_index:
        #     #     continue
        #     # TODO consider bounding final loss instead of components
        #     # Loss taken from https://arxiv.org/abs/1705.11192
        #     # Using relu to clip values below 0 since tf doesn't have a function to clip only 1 side
        #     l = tf.nn.relu(1 - self.target_energy + tf.gather(self.energy_tensor, d))
        #     self.loss_components.append(l)

        # Loss taken from https://arxiv.org/abs/1705.11192
        # Using relu to clip values below 0 since tf doesn't have a function to clip only 1 side
        loss = tf.nn.relu(1 - tf.expand_dims(self.target_energy, axis=1) + self.energy_tensor)

        self.loss = (tf.reduce_sum(loss) / self.batch_size) - 1

    def _build_optimizer(self):
        self.train_op = tf.contrib.layers.optimize_loss(
            loss=self.loss,
            global_step=tf.train.get_or_create_global_step(), # TODO define
            learning_rate=self.lr,
            optimizer="Adam",
            # some gradient clipping stabilizes training in the beginning.
            clip_gradients=self.gradient_clip)
            # summaries=["learning_rate", "loss", "gradients", "gradient_norm"])

    def fill_feed_dict(self, fd, candidates, target_idx):
        fd[self.target_indices] = target_idx
        for i, c in enumerate(candidates):
            fd[self.candidates[i]] = c

    def close(self):
        self.sess.close()

    def __del__(self):
        self.sess.close()

