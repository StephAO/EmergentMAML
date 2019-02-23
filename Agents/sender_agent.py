from Agents.agent import Agent
from data_handler import img_h, img_w
import tensorflow as tf
import tensorflow_probability as tfp

class SenderAgent(Agent):

    def __init__(self, vocab_size, num_distractors, use_images):
        self.num_fc_hidden = 256
        # TODO define temperature better - maybe learnt temperature?
        self.temperature = 5.

        super().__init__(vocab_size, num_distractors, use_images=use_images)


    def _build_input(self):
        """
        Define starting state and inputs to next state
        For sender agent:
            - starting state is output of a pre-trained model on imagenet
            - First input is a start of sentence token (sos), followed by the output of the previous timestep
        :return:
        """
        self.freeze_cnn = True
        if self.use_images:
            # Determines starting state
            self.target_image = tf.placeholder(tf.float32, shape=(self.batch_size, img_h, img_w, 3))
            self.pre_feat = Agent.pre_trained(self.target_image)
            if self.freeze_cnn:
                self.pre_feat = tf.stop_gradient(self.pre_feat)
            self.pre_feat = self.pre_feat / tf.reduce_max(self.pre_feat, axis=1, keepdims=True)

        else: # Use one-hot encoding
            self.target_image = tf.placeholder(tf.int32, shape=(self.batch_size))
            self.pre_feat = tf.one_hot(self.target_image, self.D+1)

        self.fc = tf.layers.Dense(self.num_hidden, activation=tf.nn.leaky_relu,
                                  kernel_initializer=tf.glorot_uniform_initializer)


        self.s0 = self.fc(self.pre_feat)
        # weights = tf.get_default_graph().get_tensor_by_name(os.path.split(x.name)[0] + '/kernel:0'))

        self.starting_tokens = tf.stack([self.sos_token] * self.batch_size)
        # Determines input to decoder at next time step
        # TODO: define a end_fn that actually has a chance of triggering so that we can variable len messages
        if not self.use_mlp:
            self.helper = tf.contrib.seq2seq.InferenceHelper(sample_fn=lambda outputs: outputs,
                                                             sample_shape=[self.K],
                                                             sample_dtype=tf.float32,
                                                             start_inputs=self.starting_tokens,
                                                             end_fn=lambda sample_ids:
                                                                tf.reduce_all(tf.equal(sample_ids, self.eos_token))
                                                             )


    def _build_output(self):
        # Used to map RNN output to RNN input
        if not self.use_mlp:
            output_to_input = tf.layers.Dense(self.K, kernel_initializer=tf.glorot_uniform_initializer)

            # Decoder
            self.decoder = tf.contrib.seq2seq.BasicDecoder(self.gru_cell, self.helper, initial_state=self.s0,
                                                           output_layer=output_to_input)

            self.rnn_outputs, self.final_state, self.final_sequence_lengths = \
                tf.contrib.seq2seq.dynamic_decode(self.decoder, output_time_major=True, maximum_iterations=self.L)

            self.rnn_outputs = self.rnn_outputs[0]

        else:
            # self.img_trans_1 = tf.layers.dense(self.s0, self.num_hidden, activation=tf.nn.leaky_relu,
            #                                    kernel_initializer=tf.random_normal_initializer)
            # self.img_trans_2 = tf.layers.dense(self.img_trans_1, self.num_hidden, activation=tf.nn.leaky_relu,
            #                                    kernel_initializer=tf.random_normal_initializer)
            self.rnn_outputs = tf.layers.dense(self.s0, self.K, kernel_initializer=tf.random_normal_initializer)

            self.rnn_outputs = tf.expand_dims(self.rnn_outputs, axis=0)
            self.final_sequence_lengths = None

        # self.final_features = tf.layers.dense(self.output, self.num_fc_hidden, activation=tf.nn.leaky_relu)
        # self.logits = tf.layers.dense(self.final_features, self.vocab_size, activation=None)

        # Annealing temperature
        # self.temperature = tf.clip_by_value(10000. / tf.cast(self.epoch, tf.float32), 0.5, 10.)

        self.dist = tfp.distributions.RelaxedOneHotCategorical(self.temperature, logits=self.rnn_outputs)
        self.output = self.dist.sample()
        self.output_symbol = tf.argmax(self.output, axis=2)

    def get_output(self):
        return self.output, self.output_symbol, self.final_sequence_lengths

    def fill_feed_dict(self, feed_dict, target_image):
        feed_dict[self.target_image] = target_image


