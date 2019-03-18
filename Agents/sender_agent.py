import tensorflow as tf
import tensorflow_probability as tfp

from .agent import Agent
from utils.data_handler import img_h, img_w


class SenderAgent(Agent):

    layers = []
    saver = None

    def __init__(self, straight_through=True, load_key=None, **kwargs):
        # TODO define temperature better - maybe learnt temperature?
        # with tf.variable_scope("sender"):
        self.loss = None
        self.straight_through=straight_through
        super().__init__(load_key=load_key, **kwargs)
        # Create saver
        SenderAgent.saver = SenderAgent.saver or tf.train.Saver(var_list=SenderAgent.get_weights())
        if load_key is not None:
            SenderAgent.load_model(load_key)


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
            self.target_image = tf.placeholder(tf.float32, shape=(Agent.batch_size, img_h, img_w, 3))
            self.pre_feat = Agent.pre_trained(self.target_image)
            if self.freeze_cnn:
                self.pre_feat = tf.stop_gradient(self.pre_feat)
            self.pre_feat = self.pre_feat / tf.maximum(tf.reduce_max(self.pre_feat, axis=1, keepdims=True), Agent.epsilon)

        else: # Use one-hot encoding
            self.target_image = tf.placeholder(tf.int32, shape=(Agent.batch_size))
            self.pre_feat = tf.one_hot(self.target_image, Agent.D+1)

        self.s0 = tf.nn.rnn_cell.LSTMStateTuple(self.img_fc(self.pre_feat),
                                                tf.zeros((Agent.batch_size, Agent.num_hidden), dtype=tf.float32))

        self.starting_tokens = tf.stack([self.sos_token] * Agent.batch_size)
        # Determines input to decoder at next time step
        # TODO: define a end_fn that actually has a chance of triggering so that we can variable len messages
        # TODO do this better
        self.sample_fn = lambda outputs: outputs #tf.one_hot(tf.squeeze(tf.random.multinomial(tf.log(tf.nn.softmax(outputs)), 1)), Agent.K, axis=1)
        self.next_inputs_fn = lambda outputs: tf.one_hot(tf.squeeze(tf.random.multinomial(tf.log(tf.nn.softmax(outputs)), 1)), Agent.K, axis=1)
        self.helper = tf.contrib.seq2seq.InferenceHelper(sample_fn=self.sample_fn,
                                                         sample_shape=[Agent.K],
                                                         sample_dtype=tf.float32,
                                                         start_inputs=self.starting_tokens,
                                                         end_fn=lambda sample_ids:
                                                            tf.reduce_all(tf.equal(sample_ids, self.eos_token)),
                                                         next_inputs_fn = self.next_inputs_fn
                                                         )

    def _build_output(self):
        """
        Build output of agent from the output of the RNN
        :return:
        """
        # Used to map RNN output to RNN input
        output_to_input = tf.layers.Dense(Agent.K, kernel_initializer=tf.glorot_uniform_initializer)
        SenderAgent.layers.append(output_to_input)

        # Decoder
        self.decoder = tf.contrib.seq2seq.BasicDecoder(self.rnn_cell, self.helper, initial_state=self.s0,
                                                       output_layer=output_to_input)

        self.outputs, self.final_state, self.final_sequence_lengths = \
            tf.contrib.seq2seq.dynamic_decode(self.decoder, output_time_major=True, maximum_iterations=Agent.L)
        # Select rnn_outputs from rnn_outputs: see
        # https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BasicDecoderOutput
        self.rnn_outputs = self.outputs.rnn_output

        # TODO: consider annealing temperature
        # self.temperature = tf.clip_by_value(10000. / tf.cast(self.epoch, tf.float32), 0.5, 10.)

        # Gumbel Softmax TODO: use gumbel softmax straight through used in https://arxiv.org/abs/1705.11192
        self.dist = tfp.distributions.RelaxedOneHotCategorical(Agent.temperature, logits=self.rnn_outputs)
        self.message = self.dist.sample()

        if self.straight_through:
            self.message_hard = tf.cast(tf.one_hot(tf.argmax(self.message, -1), self.K), self.message.dtype)
            self.message = tf.stop_gradient(self.message_hard - self.message) + self.message

        self.prediction = tf.argmax(self.message, axis=2, output_type=tf.int32)

    def get_output(self):
        return self.message, self.final_sequence_lengths

    def fill_feed_dict(self, feed_dict, target_image, *args):
        feed_dict[self.target_image] = target_image


