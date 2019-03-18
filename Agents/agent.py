import tensorflow as tf
import os
import sys
from utils.data_handler import img_w, img_h


class Agent(object):
    """
    Abstract base class for all agents - eventually the weights contained here will be the weights trained using MAML
    WARNING: must set parameters before creating any agents
    """
    # GAME PARAMETERS
    K = None
    D = None
    L = None  # Maximum message length

    # MODEL PARAMETERS
    freeze_cnn = True
    num_hidden = 512
    batch_size = 128
    batch_shape = (batch_size, img_h, img_w, 3)

    # TRAINING PARAMETERS
    step = tf.train.get_or_create_global_step()
    lr = 0.0005  # self._cyclicLR() #0.005
    gradient_clip = 5.0
    temperature = 5.
    loss_type = None

    # OTHER
    epsilon = 1e-12

    # Debugging so that OOM error happens on the line it is created instead of always on the session.run() call
    gpu_options = tf.GPUOptions(allow_growth=True)
    # Create session
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    #  Shared CNN pre-trained on imagenet, see https://github.com/keras-team/keras-applications for other options
    pre_trained = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet', pooling='max',
                                                                 input_shape=(img_h, img_w, 3))

    # Shared image fc layer
    img_fc = tf.keras.layers.Dense(num_hidden, activation=tf.nn.tanh,
                                   kernel_initializer=tf.glorot_uniform_initializer)
    # img_fc = tf.make_template("img_fc", img_fc)
    # img_fc.name = "shared_fc"
    # Shared GRU cell
    gru_cell = tf.nn.rnn_cell.GRUCell(num_hidden, kernel_initializer=tf.random_normal_initializer,
                                      name="shared_gru")#, reuse=tf.AUTO_REUSE)

    # list to store MAML layers
    layers = [img_fc, gru_cell]

    # Create save/load directory
    base_dir = "/h/stephaneao/EmergentMAML"#os.path.dirname(sys.modules['__main__'].__file__)
    data_dir = base_dir + '/data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    @staticmethod
    def set_params(K=None, D=None, L=None, lr=None, loss_type=None, batch_size=None, num_hidden=None, temperature=None):
        """
        Sets agent parameters. Call this before initializing any agent
        """
        Agent.K = K or Agent.K
        Agent.L = L or Agent.L
        Agent.D = D or Agent.D
        Agent.lr = lr or Agent.lr
        Agent.loss_type = loss_type or Agent.loss_type
        Agent.batch_size = batch_size or Agent.batch_size
        Agent.num_hidden = num_hidden or Agent.num_hidden
        Agent.temperature = temperature or Agent.temperature

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

    @classmethod
    def get_weights(cls):
        """
        returns a list of all weights shared by all agents
        :return:
        """
        weights = []
        for l in cls.layers:
            weights.extend(l.weights)
        return weights

    @classmethod
    def save_model(cls, exp_name):
        save_dir = Agent.data_dir + "/" + exp_name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        checkpoint_path = '/'.join([save_dir, cls.__name__, "checkpoint.ckpt"])
        cls.saver.save(Agent.sess, checkpoint_path)

    @classmethod
    def load_model(cls, exp_name):
        try:
            checkpoint_path = '/'.join([Agent.data_dir, exp_name, cls.__name__, "checkpoint.ckpt"])
            cls.saver.restore(Agent.sess, checkpoint_path)
        except tf.errors.NotFoundError:
            raise FileNotFoundError("No saved models found for experiment {}".format(exp_name))

    # Create saver
    saver = None

    def __init__(self, load_key=None, use_images=True):
        """
        Base agent, also currently holds a lot of hyper parameters
        :param vocab_size:
        :param num_distractors:
        :param use_images:
        :param loss_type:
        """
        if not (Agent.K and Agent.L and Agent.D):
            raise ValueError("Set static agent parameters using Agent.set_params() before creating any agent instances")

        # TODO deal with hyper parameters better
        self.use_images = use_images

        # TODO: properly define start/end tokens
        # Currently setting start token to [1, 0, 0, ...., 0]
        # And end token to [0, 1, 0, 0, ..., 0]
        self.sos_token = tf.one_hot(0, Agent.K)
        self.eos_token = tf.one_hot(1, Agent.K)

        # All agents need a train_op
        self.train_op = None

        # Build model
        self._build_input()
        self._build_RNN()
        self._build_output()
        self._build_losses()
        self._build_optimizer()

        # Create saver
        Agent.saver = Agent.saver or tf.train.Saver(var_list=Agent.get_weights())
        if load_key is not None:
            Agent.load_model(load_key)

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


