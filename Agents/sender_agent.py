import tensorflow as tf
import tensorflow_probability as tfp

from .agent import Agent
from utils.data_handler import img_h, img_w


class SenderAgent(Agent):
    # Used to map RNN output to RNN input
    output_to_input = None
    layers = None
    # Shared image fc layer
    img_fc = tf.keras.layers.Dense(Agent.num_hidden,
                                   kernel_initializer=tf.glorot_uniform_initializer)
    # img_fc = tf.make_template("img_fc", img_fc)
    # img_fc.name = "shared_fc"
    # Shared RNN cell
    rnn_cell = tf.nn.rnn_cell.LSTMCell(Agent.num_hidden, initializer=tf.glorot_uniform_initializer)

    # list to store MAML layers
    shared_layers = [img_fc, rnn_cell]

    saver = None

    def __init__(self, straight_through=True, load_key=None, **kwargs):
        # TODO define temperature better - maybe learnt temperature?
        # with tf.variable_scope("sender"):
        self.loss = None
        self.straight_through=straight_through
        SenderAgent.output_to_input = SenderAgent.output_to_input or \
                                      tf.layers.Dense(Agent.K, kernel_initializer=tf.glorot_uniform_initializer)
        SenderAgent.layers = [SenderAgent.output_to_input]

        super().__init__(load_key=load_key, **kwargs)
        # Create saver
        SenderAgent.saver = SenderAgent.saver or tf.train.Saver(var_list=SenderAgent.get_all_weights())
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
        self.embedding = tf.get_variable(
            name="map",
            shape=[Agent.K, Agent.emb_size],
            initializer=tf.initializers.glorot_normal)


        if self.use_images:
            # Determine starting state
            self.target_image = tf.placeholder(tf.float32, shape=(Agent.batch_size, 2048))
            # self.pre_feat = Agent.pre_trained(self.target_image)
            # if self.freeze_cnn:
            #     self.pre_feat = tf.stop_gradient(self.pre_feat)
            self.pre_feat = self.target_image / tf.maximum(tf.reduce_max(tf.abs(self.target_image), axis=1, keepdims=True), Agent.epsilon)

        else: # Use one-hot encoding
            self.target_image = tf.placeholder(tf.int32, shape=(Agent.batch_size))
            self.pre_feat = tf.one_hot(self.target_image, Agent.D+1)


        self.pre_feat = self.img_fc(self.pre_feat)
        self.L_pre_feat = tf.keras.layers.RepeatVector(self.L)(self.pre_feat)
        self.s0 = tf.nn.rnn_cell.LSTMStateTuple(self.pre_feat, self.pre_feat)

        self.starting_tokens = tf.stack([Agent.V.sos_id] * Agent.batch_size)
        # Determines input to decoder at next time step
        # TODO: define a end_fn that actually has a chance of triggering so that we can variable len messages
        # TODO do this better
        # self.sample_fn = lambda outputs: outputs #tf.one_hot(tf.squeeze(tf.random.multinomial(tf.log(tf.nn.softmax(outputs)), 1)), Agent.K, axis=1)
        # self.next_inputs_fn = lambda outputs: tf.one_hot(tf.squeeze(tf.random.multinomial(tf.log(tf.nn.softmax(outputs)), 1)), Agent.K, axis=1)
        # self.helper = tf.contrib.seq2seq.InferenceHelper(sample_fn=self.sample_fn,
        #                                                  sample_shape=[Agent.K],
        #                                                  sample_dtype=tf.float32,
        #                                                  start_inputs=self.starting_tokens,
        #                                                  end_fn=lambda sample_ids:
        #                                                     tf.reduce_all(tf.equal(sample_ids, self.eos_token)),
        #                                                  next_inputs_fn = self.next_inputs_fn
        #                                                  )


        self.helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding, self.starting_tokens, Agent.V.eos_id)

        # TODO add attention?
        # self.attention = tf.contrib.seq2seq.BahdanauAttention(Agent.num_hidden, self.pre_feat, self.L)
        #
        # SenderAgent.rnn_cell = tf.contrib.seq2seq.AttentionWrapper(
        #     cell=SenderAgent.rnn_cell,
        #     attention_mechanism=self.attention,
        #     attention_layer_size=Agent.num_hidden,
        #     cell_input_fn= tf.keras.layers.Dense(Agent.num_hidden)(self.input),
        #     initial_cell_state=self.pre_feat,
        #     alignment_history=False,
        #     name='Attention_Wrapper')



    def _build_output(self):
        """
        Build output of agent from the output of the RNN
        :return:
        """

        # Decoder
        self.decoder = tf.contrib.seq2seq.BasicDecoder(self.rnn_cell, self.helper, initial_state=self.s0)
                                                       # output_layer=SenderAgent.output_to_input)



        self.outputs, self.final_state, self.final_sequence_lengths = \
            tf.contrib.seq2seq.dynamic_decode(self.decoder, maximum_iterations=Agent.L) #output_time_major=True,

        # outputs, state = tf.nn.dynamic_rnn(cell=SenderAgent.rnn_cell,
        #                                     inputs=self.input,
        #                                     # sequence_length=sequence_length,
        #                                     initial_state=self.s0,
        #                                     dtype=tf.float32)



        # Select rnn_outputs from rnn_outputs: see
        # https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BasicDecoderOutput
        # print(outputs.shape)
        # print(self.outputs.rnn_output.shape)
        self.rnn_outputs = SenderAgent.output_to_input(self.outputs.rnn_output)

        # TODO: consider annealing temperature
        # self.temperature = tf.clip_by_value(10000. / tf.cast(self.epoch, tf.float32), 0.5, 10.)

        self.dist = tfp.distributions.RelaxedOneHotCategorical(Agent.temperature, logits=self.rnn_outputs)
        self.message = self.dist.sample()

        if self.straight_through:
            self.message_hard = tf.cast(tf.one_hot(tf.argmax(self.message, -1), self.K), self.message.dtype)
            self.message = tf.stop_gradient(self.message_hard - self.message) + self.message
            # self.message =
            # print(self.message.shape)
            # print(self.embedding.shape)

        self.prediction = tf.argmax(tf.nn.softmax(self.rnn_outputs), axis=2, output_type=tf.int32)

    def get_output(self):
        return self.message, self.final_sequence_lengths

    def fill_feed_dict(self, feed_dict, target_image, *args):
        feed_dict[self.target_image] = target_image


