import tensorflow as tf
import tensorflow_probability as tfp

from Agents.agent import Agent
from utils.data_handler import img_h, img_w


class SenderAgent(Agent):

    layers = []

    def __init__(self, *args, **kwargs):
        # TODO define temperature better - maybe learnt temperature?
        # with tf.variable_scope("sender"):
        self.loss = None
        super().__init__(*args, **kwargs)


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

        self.s0 = self.img_fc(self.pre_feat)

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
        self.decoder = tf.contrib.seq2seq.BasicDecoder(self.gru_cell, self.helper, initial_state=self.s0,
                                                       output_layer=output_to_input)

        self.rnn_outputs, self.final_state, self.final_sequence_lengths = \
            tf.contrib.seq2seq.dynamic_decode(self.decoder, output_time_major=True, maximum_iterations=Agent.L)

        # Select rnn_outputs from rnn_outputs: see
        # https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BasicDecoderOutput
        self.rnn_outputs = self.rnn_outputs[0]

        # TODO: consider annealing temperature
        # self.temperature = tf.clip_by_value(10000. / tf.cast(self.epoch, tf.float32), 0.5, 10.)

        # Gumbel Softmax TODO: use gumbel softmax straight through used in https://arxiv.org/abs/1705.11192
        self.dist = tfp.distributions.RelaxedOneHotCategorical(Agent.temperature, logits=self.rnn_outputs)
        self.message = self.dist.sample()
        self.prediction = tf.argmax(self.message, axis=2, output_type=tf.int32)

    def set_loss(self, loss):
        self.loss = loss
        self._build_optimizer()

    def _build_optimizer(self):
        if self.loss is None:
            return None

        # TODO test whether having double agent weights (here and in receiver) has an affect)
        self.train_op = tf.contrib.layers.optimize_loss(
            loss=self.loss,
            global_step=Agent.step,
            learning_rate=Agent.lr,
            optimizer="Adam",
            # some gradient clipping stabilizes training in the beginning.
            clip_gradients=Agent.gradient_clip,
            # only update sender agent weights
            variables=Agent.get_weights() + SenderAgent.get_weights()
        )

        # Initialize
        Agent.sess.run(tf.global_variables_initializer())

    def get_output(self):
        return self.message, self.final_sequence_lengths

    def fill_feed_dict(self, feed_dict, target_image, *args):
        feed_dict[self.target_image] = target_image


