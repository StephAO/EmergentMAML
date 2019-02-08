from data_handler import img_w, img_h
import numpy as np
import tensorflow as tf

class Agent:

    # pre_trained = tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', pooling='max',
    #                                                       input_shape=(img_h, img_w, 3))
    def __init__(self, vocab_size, num_distractors):
        self.num_hidden = 256
        self.batch_size = 64
        self.batch_shape = (self.batch_size, img_h, img_w, 3)
        self.K = vocab_size
        self.D = num_distractors
        self.max_len = tf.constant(15) # maximum message length
        self.lr = 0.001
        self.gradient_clip = 10.0
        # TODO: properly define start/end tokens
        # Currently setting start token to [1, 0, 0, ...., 0]
        # And end token to [0, 1, 0, 0, ..., 0]
        self.sos_token = tf.one_hot(0, self.K)
        self.eos_token = tf.one_hot(1, self.K)


        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


        self._build_input()
        self._build_RNN()
        self._build_output()
        self._build_losses()
        self._build_optimizer()

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
        self.gru_cell = tf.nn.rnn_cell.GRUCell(self.num_hidden)



    def _build_output(self):
        """
        Define output from RNN
        :return: None
        """
        pass

    def _build_losses(self):
        pass

    def _build_optimizer(self):
        pass

    def get_output(self):
        return self.output

    def run_game(self, fd):
        return self.sess.run([self.train_op, self.human_msg, self.output, self.loss, self.target_energy], feed_dict=fd)[1:]

