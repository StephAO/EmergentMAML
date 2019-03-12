from Agents.agent import Agent
from data_handler import img_h, img_w
import tensorflow as tf
import tensorflow_probability as tfp

class SenderAgent(Agent):

    def __init__(self, vocab_size, num_distractors, use_images):
        # TODO define temperature better - maybe learnt temperature?
        self.temperature = 5.
        super().__init__(vocab_size, num_distractors, use_images=use_images)


    def _build_input(self):
        """
        Build starting state and inputs to next state
        For sender agent:
            - Starting state is output of a pre-trained model on imagenet
            - First input is a start of sentence token (sos), followed by the output of the previous timestep
        :return:
        """
        if self.use_images:
            # Determine starting state
            self.target_image = tf.placeholder(tf.float32, shape=(self.batch_size, img_h, img_w, 3))
            self.pre_feat = Agent.pre_trained(self.target_image)
            if self.freeze_cnn:
                self.pre_feat = tf.stop_gradient(self.pre_feat)
            self.pre_feat = self.pre_feat / tf.maximum(tf.reduce_max(self.pre_feat, axis=1, keepdims=True), self.epsilon)

        else: # Use one-hot encoding
            self.target_image = tf.placeholder(tf.int32, shape=(self.batch_size))
            self.pre_feat = tf.one_hot(self.target_image, self.D+1)

        self.s0 = Agent.img_fc(self.pre_feat)

        self.starting_tokens = tf.stack([self.sos_token] * self.batch_size)
        # Determines input to decoder at next time step
        # TODO: define a end_fn that actually has a chance of triggering so that we can variable len messages
        self.helper = tf.contrib.seq2seq.InferenceHelper(sample_fn=lambda outputs: outputs,
                                                         sample_shape=[self.K],
                                                         sample_dtype=tf.float32,
                                                         start_inputs=self.starting_tokens,
                                                         end_fn=lambda sample_ids:
                                                            tf.reduce_all(tf.equal(sample_ids, self.eos_token))
                                                         )

    def _build_output(self):
        """
        Build output of agent from the output of the RNN
        :return:
        """
        # Used to map RNN output to RNN input
        output_to_input = tf.layers.Dense(self.K, #activation=lambda x: tf.nn.softmax,
                                          kernel_initializer=tf.glorot_uniform_initializer)

        # Decoder
        self.decoder = tf.contrib.seq2seq.BasicDecoder(Agent.gru_cell, self.helper, initial_state=self.s0,
                                                       output_layer=output_to_input)

        self.rnn_outputs, self.final_state, self.final_sequence_lengths = \
            tf.contrib.seq2seq.dynamic_decode(self.decoder, output_time_major=True, maximum_iterations=self.L)

        # Select rnn_outputs from rnn_outputs: see
        # https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BasicDecoderOutput
        self.rnn_outputs = self.rnn_outputs[0]

        # TODO: consider annealing temperature
        # self.temperature = tf.clip_by_value(10000. / tf.cast(self.epoch, tf.float32), 0.5, 10.)

        # Gumbel Softmax TODO: use gumbel softmax straight through used in https://arxiv.org/abs/1705.11192
        self.dist = tfp.distributions.RelaxedOneHotCategorical(self.temperature, logits=self.rnn_outputs)
        self.output = self.dist.sample()
        self.prediction = tf.argmax(self.output, axis=2, output_type=tf.int32)

    def get_output(self):
        return self.output, self.prediction, self.final_sequence_lengths

    def fill_feed_dict(self, feed_dict, target_image):
        feed_dict[self.target_image] = target_image


