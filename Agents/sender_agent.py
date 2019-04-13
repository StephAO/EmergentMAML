import tensorflow as tf
import tensorflow_probability as tfp

from .agent import Agent

class SenderAgent(Agent):
    rnn_cell = tf.nn.rnn_cell.LSTMCell(Agent.num_hidden)
    # Unique fc rnn output layer
    output_to_input = None
    layers = []

    # Shared embedding and image fc layer
    embedding = None
    img_fc = tf.keras.layers.Dense(Agent.num_hidden)
    shared_layers = [img_fc]

    if Agent.split_sr:
        layers += [rnn_cell]
    else:
        shared_layers += [rnn_cell]

    saver = None
    loaded = False

    def __init__(self, load_key=None):
        # Setup rnn output to input layer and embedding if it hasn't already been done
        if SenderAgent.output_to_input is None:
            SenderAgent.output_to_input = tf.layers.Dense(Agent.K)
            SenderAgent.layers += [SenderAgent.output_to_input]
        if SenderAgent.embedding is None:
            SenderAgent.embedding = tf.get_variable(name="map", shape=[Agent.K, Agent.emb_size])
            SenderAgent.shared_layers += [SenderAgent.embedding]

        super().__init__()
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
              concatenated with the imgae features
        """
        # Determine starting state
        self.target_image = tf.placeholder(tf.float32, shape=(Agent.batch_size, 2048))
        self.pre_feat = self.target_image / tf.maximum(tf.reduce_max(tf.abs(self.target_image), axis=1, keepdims=True), Agent.epsilon)

        self.pre_feat = self.img_fc(self.pre_feat)
        # self.L_pre_feat = tf.keras.layers.RepeatVector(self.L)(self.pre_feat)
        self.s0 = tf.nn.rnn_cell.LSTMStateTuple(self.pre_feat, tf.zeros((Agent.batch_size, Agent.num_hidden)))
        # self.bsd_s0 = tf.contrib.seq2seq.tile_batch(self.s0, multiplier=Agent.beam_width)

        # Create decoding helper
        self.starting_tokens = tf.stack([Agent.V.sos_id] * Agent.batch_size)
        self.starting_embed = tf.nn.embedding_lookup(SenderAgent.embedding, self.starting_tokens)
        # self.starting_combined = tf.concat((self.starting_embed, self.pre_feat), axis=1)
        # Determines input to decoder at next time step
        # Copying greedy helper, taken from https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/seq2seq/python/ops/helper.py
        sample_fn = lambda outputs: tf.argmax(outputs, axis=-1, output_type=tf.int32)
        def next_inputs_fn(sample_ids):
            embed = tf.nn.embedding_lookup(SenderAgent.embedding, sample_ids)
            # combined = tf.concat((embed, self.pre_feat), axis=1)
            return embed

        self.helper = tf.contrib.seq2seq.InferenceHelper(sample_fn=sample_fn,
                                                         sample_shape=[],
                                                         sample_dtype=tf.int32,
                                                         start_inputs=self.starting_embed,
                                                         end_fn=lambda sample_ids: tf.equal(sample_ids, Agent.V.eos_id),
                                                         next_inputs_fn = next_inputs_fn
                                                         )

        # self.helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(SenderAgent.embedding, self.starting_tokens, Agent.V.eos_id)

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
        Build output of agent
        """
        # Decoder
        # if Agent.train:
        self.decoder = tf.contrib.seq2seq.BasicDecoder(self.rnn_cell, self.helper, initial_state=self.s0)
        # else:
        #     self.decoder = tf.contrib.seq2seq.BeamSearchDecoder(self.rnn_cell, SenderAgent.embedding,
        #                                                         self.starting_tokens, Agent.V.eos_id,
        #                                                         self.bsd_s0, beam_width=Agent.beam_width,
        #                                                         output_layer=SenderAgent.output_to_input)

        self.outputs, self.final_state, self.final_sequence_lengths = \
            tf.contrib.seq2seq.dynamic_decode(self.decoder, maximum_iterations=Agent.L)
        # Select rnn_outputs from rnn_outputs: see
        # https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BasicDecoderOutput
        # if Agent.train:
        self.rnn_outputs = SenderAgent.output_to_input(self.outputs.rnn_output)


        # TODO: consider annealing temperature
        # self.temperature = tf.clip_by_value(10000. / tf.cast(self.epoch, tf.float32), 0.5, 10.)

        self.dist = tfp.distributions.RelaxedOneHotCategorical(Agent.temperature, logits=self.rnn_outputs)
        self.message = self.dist.sample()

        if Agent.straight_through:
            self.message_hard = tf.cast(tf.one_hot(tf.argmax(self.message, -1), self.K), self.message.dtype)
            self.message = tf.stop_gradient(self.message_hard - self.message) + self.message

        self.prediction = tf.argmax(tf.nn.softmax(self.rnn_outputs), axis=2, output_type=tf.int32)
        self.probabilities = tf.nn.softmax(self.rnn_outputs)

        # else:
        #     self.probabilities = self.outputs.predicted_ids

    def get_output(self):
        return self.message, self.final_sequence_lengths

    def fill_feed_dict(self, feed_dict, target_image):
        """
        Fill necessary placeholders required for sender
        :param feed_dict: feed_dict to fill
        :param target_image: image to fill placeholder with
        """
        feed_dict[self.target_image] = target_image

