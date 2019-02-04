from agent import Agent
import tensorflow as tf

class ReceiverAgent(Agent):

    def __init__(self, message, target_image):
        super().__init__()
        self.query_key_size = 128
        self.message = message
        self.target_image = target_image

    def _build_input(self):
        """
        Define starting state and inputs to next state
        For receiver agent:
            - starting state is the zero state.
            - First input is a start of sentence token (sos), followed by the output of the previous timestep
        :return: None
        """
        # TODO: define a better starting state for receiver
        self.s0 = tf.zeros(self.batch_size, self.num_hidden)
        self.helper = tf.contrib.seq2seq.TrainingHelper(self.message, self.max_len)

    def _build_output(self):
        """
        Take output of RNN, h_t, pass it through a one layer perceptron, g(h_t).
        Pass each image, i_k, through a pre-trained model, then a one layer perceptron, f(i_k)
        Use -f(i_k)^T g(h_t) as an energy function
        :return: chosen index of chosen image
        """
        super()._build_output()
        # Get RNN features
        self.RNN_features = tf.layers.dense(self.output, self.query_key_size, activation=tf.nn.relu)

        # Get image features
        # TODO - can we do this with only matrices?
        self.candidates = []
        self.image_features = []
        self.energies = []
        for d in range(self.D):
            self.candidates.append(tf.placeholder(tf.float32))
            img_feat = self.pre_trained(self.candidates)
            img_feat = tf.layers.dense(img_feat, self.query_key_size, activation=tf.nn.relu)
            self.image_features.append(img_feat)

            self.energies.append(tf.negative(tf.matmul(tf.transpose(img_feat), self.RNN_features)))

        self.energy_tensor = tf.concat(self.energies)

        self.prob_dist = tf.nn.softmax(self.energy_tensor)

        self.output = tf.argmax(self.prob_dist)

    def _build_losses(self):
        self.target_index = tf.placeholder(tf.int32)
        self.target_energy = self.energies[self.target_index]
        self.loss_components = []
        for d in range(self.D):
            if d == self.target_index:
                continue
            # TODO consider bounding final loss instead of components
            # Loss taken from https://arxiv.org/abs/1705.11192
            self.loss_components.append(
                tf.reduce_max(0, 1 - self.target_energy + self.energies[d])
            )
        self.loss = tf.reduce_sum(tf.concat(self.loss_components))









