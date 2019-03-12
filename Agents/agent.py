from data_handler import img_w, img_h
import numpy as np
import tensorflow as tf

class Agent:
    """
    Abstract base class for all agents - eventually the weights contained here will be the weights trained using MAML
    """
    # GAME PARAMETERS
    K = None
    D = None
    L = 3  # Maximum message length

    # MODEL PARAMETERS
    freeze_cnn = True
    num_hidden = 512
    batch_size = 64
    batch_shape = (batch_size, img_h, img_w, 3)

    # TRAINING PARAMETERS
    step = tf.train.get_or_create_global_step()
    lr = 0.001  # self._cyclicLR() #0.005
    gradient_clip = 10.0
    temperature = 5.
    loss_type = None

    # OTHER
    epsilon = 1e-12

    with tf.variable_scope("MAML"):
    #  Shared CNN pre-trained on imagenet, see https://github.com/keras-team/keras-applications for other options
        pre_trained = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet', pooling='max',
                                                                     input_shape=(img_h, img_w, 3))
        # Shared image fc layer
        img_fc = tf.keras.layers.Dense(num_hidden, activation=tf.nn.tanh,
                                       kernel_initializer=tf.glorot_uniform_initializer)

        # Shared GRU cell
        gru_cell = tf.nn.rnn_cell.GRUCell(num_hidden, kernel_initializer=tf.random_normal_initializer)

    @staticmethod
    def get_params():
        """
        Returns a dictionary of parameters to track.
        """
        params = {
            "K": Agent.K,
            "D": Agent.D,
            "L": Agent.L,
            "lr": Agent.lr,
            "loss_type": Agent.loss_type,
            "batch_size": Agent.batch_size,
            "num_hidden": Agent.num_hidden,
            "temperature": Agent.temperature
        }

        return params

    def __init__(self, vocab_size, num_distractors, use_images=False, loss_type='pairwise', freeze_cnn=True,
                 track_results=True):
        """
        Base agent, also currently holds a lot of hyper parameters
        :param vocab_size:
        :param num_distractors:
        :param use_images:
        :param loss_type:
        """
        # TODO deal with hyper parameters better
        # GAME PARAMETERS
        Agent.K = vocab_size
        Agent.D = num_distractors
        self.use_images = use_images

        # MODEL PARAMETERS
        Agent.freeze_cnn = freeze_cnn

        # TRAINING PARAMETERS
        Agent.loss_type = loss_type
        self.step = tf.train.get_or_create_global_step()

        # TODO: properly define start/end tokens
        # Currently setting start token to [1, 0, 0, ...., 0]
        # And end token to [0, 1, 0, 0, ..., 0]
        self.sos_token = tf.one_hot(0, self.K)
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

    def run_game(self, fd, data_type="train"):
        ops = [self.message, self.accuracy, self.loss, self.step]
        if data_type == "train":
            ops += [self.train_op]
        return self.sess.run(ops, feed_dict=fd)[:4]

