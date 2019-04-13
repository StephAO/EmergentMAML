import tensorflow as tf
import os
import sys
from utils.data_handler import project_path
from utils.vocabulary import Vocabulary as V

def get_list_of_variables(lst):
    """
    helper function to get a list of weights given a list of keras layers and tf variables
    :param lst: list of keras layers and tf variables
    :return: list of tf variables
    """
    weights = []
    for l in lst:
        if isinstance(l, tf.Variable):
            weights.append(l)
        else:
            weights.extend(l.weights)
    return weights


class Agent(object):
    """
    Abstract base class for all agents
    WARNING: must set parameters before creating any agents
    """
    # GAME PARAMETERS
    K = None  # Vocabulary size
    D = None  # Number of distractors
    L = None  # Maximum message length

    # MODEL PARAMETERS
    freeze_cnn = False
    num_hidden = 512
    batch_size = 128
    emb_size = 300

    # TRAINING PARAMETERS
    step = tf.train.get_or_create_global_step()
    lr = 0.0005  # self._cyclicLR() #0.005
    gradient_clip = 5.0
    # TODO define temperature better - maybe learnt temperature?
    temperature = 1.
    loss_type = None
    straight_through = True
    train = True
    split_sr = True # if true Sender and receiver have separate rnns during reptile training

    # OTHER
    epsilon = 1e-12
    beam_width = 3

    # Debugging so that OOM error happens on the line it is created instead of always on the session.run() call
    gpu_options = tf.GPUOptions(allow_growth=True)
    # Create session
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # Create save/load directory
    base_dir = project_path
    data_dir = base_dir + '/data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Set up vocabulary
    V = None

    @staticmethod
    def set_params(K=None, D=None, L=None, lr=None, train=None, loss_type=None, batch_size=None, num_hidden=None,
                   temperature=None):
        """
        Sets agent parameters. Call this before initializing any agent
        """
        Agent.K = K or Agent.K
        Agent.L = L or Agent.L
        Agent.D = D or Agent.D
        Agent.lr = lr or Agent.lr
        Agent.train = train if train is not None else Agent.train
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
            "split_sr": Agent.split_sr,
            "loss_type": Agent.loss_type,
            "batch_size": Agent.batch_size,
            "num_hidden": Agent.num_hidden,
            "temperature": Agent.temperature,
            "embedding_size": Agent.emb_size,
            "straight_through": Agent.straight_through
        }

        return params

    @classmethod
    def get_weights(cls):
        """ Returns a list of all weights unique to agent"""
        return get_list_of_variables(cls.layers)

    @classmethod
    def get_shared_weights(cls):
        """ Returns a list of all weights shared by all agents"""
        return get_list_of_variables(cls.shared_layers)

    @classmethod
    def get_all_weights(cls):
        """ Returns a list of all agent weights"""
        return get_list_of_variables(cls.layers + cls.shared_layers)

    @classmethod
    def save_model(cls, exp_name):
        if cls.saver is None:
            return
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
            print("Loaded weights for {}".format(cls.__name__))
            cls.loaded = True
        except tf.errors.NotFoundError:
            raise FileNotFoundError("No saved models found for experiment {}".format(exp_name))

    def __init__(self):
        """Base agent, also currently holds a lot of hyper parameters"""
        if not (Agent.K and Agent.L and Agent.D):
            raise ValueError("Set static agent parameters using Agent.set_params() before creating any agent instances")

        # Create vocabulary
        if Agent.V is None:
            Agent.V = V()
            try:
                Agent.V.load_vocab()
            except FileNotFoundError:
                Agent.V.generate_vocab()
                Agent.V.save_vocab()

            Agent.V.generate_top_k(Agent.K)

        # All agents need a train_op
        self.train_op = None

        # Build model
        self._build_input()
        self._build_output()
        # if Agent.train:
        self._build_losses()
        self._build_optimizer()

    def _cyclicLR(self):
        """
        Cyclical triangular learning rate Based on https://github.com/bckenstler/CLR
        Currently unused
        :return: cyclic learning rate
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
        """
        pass

    def _build_output(self):
        """
        Abstract Method
        Build output of agent from the output of the RNN
        """
        pass

    def _build_losses(self):
        """
        Abstract Method
        Build losses for agent
        """
        pass

    def _build_optimizer(self):
        """
        Abstract Method
        Build optimizer and train_op of agent
        """
        pass

    def close(self):
        Agent.sess.close()

    def __del__(self):
        Agent.sess.close()
