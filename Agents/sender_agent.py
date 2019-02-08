from Agents.agent import Agent
from data_handler import img_h, img_w
import tensorflow as tf
import tensorflow_probability as tfp

class SenderAgent(Agent):

    def __init__(self, vocab_size, num_distractors):
        self.num_fc_hidden = 128
        # TODO define temperature better - maybe learnt temperature?
        self.temperature = 5.
        self.L = 1 # Maximum number of iterations
        super().__init__(vocab_size, num_distractors)

    def _build_input(self):
        """
        Define starting state and inputs to next state
        For sender agent:
            - starting state is output of a pre-trained model on imagenet
            - First input is a start of sentence token (sos), followed by the output of the previous timestep
        :return:
        """
        # Determines starting state
        # self.target_image = tf.placeholder(tf.float32, shape=(self.batch_size, img_h, img_w, 3))
        # img_feat = Agent.pre_trained(self.target_image)

        ### TEST
        self.target_image = tf.placeholder(tf.int32, shape=(self.batch_size))
        img_feat = tf.one_hot(self.target_image, self.K)
        ###

        self.fc = tf.layers.Dense(self.num_hidden, activation=tf.nn.relu,
                                  kernel_initializer=tf.glorot_uniform_initializer)
        self.s0 = self.fc(img_feat)
        self.fc_weights_avg = tf.reduce_max(self.fc.weights[0])
        # weights = tf.get_default_graph().get_tensor_by_name(os.path.split(x.name)[0] + '/kernel:0'))

        self.starting_tokens = tf.stack([self.sos_token] * self.batch_size)
        # Determines input to decoder at next time step
        self.helper = tf.contrib.seq2seq.InferenceHelper(sample_fn=lambda outputs: outputs,
                                                         sample_shape=[self.K],
                                                         sample_dtype=tf.float32,
                                                         start_inputs=self.starting_tokens,
                                                         end_fn=lambda sample_ids:
                                                            tf.reduce_all(tf.equal(sample_ids, self.eos_token))
                                                         )


    def _build_output(self):
        # Used to map RNN output to RNN input
        output_to_input = tf.layers.Dense(self.K, kernel_initializer=tf.glorot_uniform_initializer)

        # Decoder
        self.decoder = tf.contrib.seq2seq.BasicDecoder(self.gru_cell, self.helper, initial_state=self.s0,
                                                       output_layer=output_to_input)

        self.rnn_outputs, self.final_state, self.final_sequence_lengths = \
            tf.contrib.seq2seq.dynamic_decode(self.decoder, output_time_major=True, maximum_iterations=self.L)

        self.rnn_outputs = self.rnn_outputs[0]

        # self.final_features = tf.layers.dense(self.output, self.num_fc_hidden, activation=tf.nn.relu)
        # self.logits = tf.layers.dense(self.final_features, self.vocab_size, activation=None)
        self.dist = tfp.distributions.RelaxedOneHotCategorical(self.temperature, logits=self.rnn_outputs)
        self.output = self.dist.sample()
        self.output_symbol = tf.argmax(self.output, axis=2)

    def get_output(self):
        return self.output, self.output_symbol, self.final_sequence_lengths

    def fill_feed_dict(self, feed_dict, target_image):
        feed_dict[self.target_image] = target_image


