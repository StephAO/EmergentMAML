from data_handler import img_w, img_h
import numpy as np
import tensorflow as tf

class Agent:
    """
    Abstract base class for all agents - eventually the weights contained here will be the weights trained using MAML
    """

    #  Shared CNN pre-trained on imagenet, see https://github.com/keras-team/keras-applications for other options
    pre_trained = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet', pooling='max',
                                                                 input_shape=(img_h, img_w, 3))
    # Shared image fc layer
    img_fc = tf.keras.layers.Dense(512, activation=tf.nn.tanh,
                                   kernel_initializer=tf.glorot_uniform_initializer)

    # Shared GRU cell
    gru_cell = tf.nn.rnn_cell.GRUCell(512, kernel_initializer=tf.random_normal_initializer)

    def __init__(self, vocab_size, num_distractors, use_images=False, loss_type='pairwise', freeze_cnn=True):
        """
        Base agent, also currently holds a lot of hyper parameters
        :param vocab_size:
        :param num_distractors:
        :param use_images:
        :param loss_type:
        """
        # TODO deal with hyper parameters better
        # GAME PARAMETERS
        self.K = vocab_size
        self.D = num_distractors
        self.L = 1  # Maximum message length
        self.use_images = use_images

        # MODEL PARAMETERS
        self.freeze_cnn = freeze_cnn
        self.num_hidden = 512
        self.batch_size = 16
        self.batch_shape = (self.batch_size, img_h, img_w, 3)

        # TRAINING PARAMETERS
        self.epoch = tf.train.get_or_create_global_step()
        self.lr = 0.001 #self._cyclicLR() #0.005
        self.gradient_clip = 10.0
        self.loss_type = loss_type

        # OTHER
        self.epsilon = 1e-12

        # TODO: properly define start/end tokens
        # Currently setting start token to [1, 0, 0, ...., 0]
        # And end token to [0, 1, 0, 0, ..., 0]
        self.sos_token = tf.ones((self.K,))
        self.eos_token = tf.one_hot(1, self.K)

        # Debugging so that OOM error happens on the line it is created instead of always on the session.run() call
        gpu_options = tf.GPUOptions(allow_growth=True)
        # Create session
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # Build model
        self._build_input()
        self._build_RNN()
        self._build_output()
        self._build_losses()
        self._build_optimizer()

        # Initialize
        self.sess.run(tf.global_variables_initializer())

    def _cyclicLR(self):
        """
        Cyclical triangular learning rate Based on https://github.com/bckenstler/CLR
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
        Build starting state and inputs to next state
        :return: None
        """
        pass

    def _build_RNN(self):
        """
        Build main RNN of agent
        :return: None
        """
        # TODO Use tf.contrib.cudnn_rnn.CudnnGRU for better GPU performance
        # TODO test GRU vs LSTM
        # self.gru_cell = tf.nn.rnn_cell.GRUCell(self.num_hidden, kernel_initializer=tf.random_normal_initializer)
        pass

    def _build_output(self):
        """
        Abstract Method
        Build output of agent from the output of the RNN
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

