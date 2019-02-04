import numpy as np
import tensorflow as tf

class Agent:

    def __init__(self, data_iterator, vocab_size, num_distractors):
        self.hidden_size = 512
        self.input_size = 256
        self.batch_size = 1
        self.vocab_size = vocab_size
        self.data_iterator = data_iterator
        self.D = num_distractors
        self.max_len = 15 # maximum message length
        self.lr = 0.0001
        self.gradient_clip = 9.0
        # TODO: properly define start/end tokens
        # Currently setting start token to [1, 0, 0, ...., 0]
        # And end token to [0, 1, 0, 0, ..., 0]
        self.sos_token = tf.one_hot(0, self.vocab_size)
        self.eos_token = tf.one_hot(1, self.vocab_size)

        self.pre_trained = tf.keras.applications.xception.Xception(include_top=True, weights='imagenet')

        self._build_input()
        self._build_RNN()
        self._build_output()
        self._build_losses()
        self._build_optimizer()

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

    def _build_input(self):
        """
        Abstract Method
        Define starting state and inputs to next state
        :return: None
        """
        pass

    def _build_RNN(self):
        """
        Build main RNN of agent
        :return: None
        """
        # TODO Use tf.contrib.cudnn_rnn.CudnnGRU for better GPU performance
        decoder_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)

        # Used to map RNN output to RNN input
        output_to_input = tf.layers.Dense(self.input_size)

        # Decoder
        self.decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, self.helper, initial_state=self.s0,
                                                       output_layer=output_to_input)

    def _build_output(self):
        """
        Define output from RNN
        :return: None
        """
        self.output = self.decoder

    def _build_losses(self):
        pass

    def _build_optimizer(self):
        self.train_op = tf.contrib.layers.optimize_loss(
            loss=self.loss,
            #global_step=tf.train.get_global_step(), # TODO define
            learning_rate=self.lr,
            optimizer="Adam",
            # some gradient clipping stabilizes training in the beginning.
            clip_gradients=self.gradient_clip)
            # summaries=["learning_rate", "loss", "gradients", "gradient_norm"])

    def get_output(self):
        return self.output

    def run_game(self, target, distractors):
        target_idx = np.random.randint(self.D + 1)
        candidates = distractors[:target_idx] + [target] + distractors[target_idx:]
        fd = {self.target_image: target, self.target_index: target_idx}
        for i, c in enumerate(candidates):
            fd[self.candidates[i]] = c

        output = self.sess.run([self.train_op, self.message, self.prediction, self.loss], feed_dict=fd)
    
        return output[1:]

    def close(self):
        self.sess.close()

    def __del__(self):
        self.sess.close()