from data_handler import img_w, img_h
import numpy as np
import tensorflow as tf

class Agent:

    pre_trained = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet', pooling='max',
                                                                 input_shape=(img_h, img_w, 3))

    def __init__(self, vocab_size, num_distractors, use_images=False, loss_type='pairwise'):
        """
        Base agent, also currently holds a lot of hyper parameters
        :param vocab_size:
        :param num_distractors:
        :param use_images:
        :param loss_type:
        """
        # TODO deal with hyper parameters better
        self.use_images = use_images
        self.num_hidden = 512
        self.batch_size = 32
        self.batch_shape = (self.batch_size, img_h, img_w, 3)
        self.K = vocab_size
        self.D = num_distractors
        self.L = 1  # Maximum number of iterations
        self.max_len = tf.constant(15) # maximum message length
        self.epoch = tf.train.get_or_create_global_step()
        self.lr = 0.0001 #self._cyclicLR() #0.005
        self.gradient_clip = 10.0
        self.loss_type = loss_type
        # TODO: properly define start/end tokens
        # Currently setting start token to [1, 0, 0, ...., 0]
        # And end token to [0, 1, 0, 0, ..., 0]
        self.sos_token = tf.ones((self.K,))
        self.eos_token = tf.one_hot(1, self.K)


        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


        self._build_input()
        self._build_RNN()
        self._build_output()
        self._build_losses()
        self._build_optimizer()

        self.sess.run(tf.global_variables_initializer())

    def _cyclicLR(self):
        """
        Based on https://github.com/bckenstler/CLR
        :return:
        """
        step_size = 100
        base_lr = 0.0001
        max_lr = 0.001

        cycle = tf.floor(1 + self.epoch / (2 * step_size))
        x = tf.abs(self.epoch / step_size - 2 * cycle + 1)
        return base_lr + (max_lr - base_lr) * tf.nn.relu(1 - x) / tf.cast((2 ** (cycle - 1)), tf.float64)

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
        self.gru_cell = tf.nn.rnn_cell.GRUCell(self.num_hidden, kernel_initializer=tf.random_normal_initializer)



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
        return self.sess.run([self.train_op, self.message, self.output, self.loss], feed_dict=fd)[1:]

