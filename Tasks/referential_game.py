import numpy as np
from comet_ml import Experiment
import matplotlib.pyplot as plt
from utils.vocabulary import Vocabulary as V

from Agents import Agent
from Tasks import Task


class ReferentialGame(Task):
    """
    Class to play referential game, where a sender sees a target image and must send a message to a receiver, who must
    the pick the target image from a set of candidate images.
    Attributes:
        Sender [Agent]
        Receiver [Agent]
        Vocabulary Size [Int]
        Distractor Set Size [Int]
    """
    def __init__(self, Sender, Receiver, **kwargs):
        """
        :param K [Int]: Vocabulary Size
        :param D [Int]: Distractor Set Size
        :param use_images[Bool]: Whether to use images or one hot encoding (for debugging)
        """
        self.sender = Sender
        self.receiver = Receiver
        super().__init__([Sender, Receiver], **kwargs)
        self.name = "Referential Game"

    def train_batch(self, inputs, mode="train"):
        """
        Play a single instance of the game
        :return:
        """
        images = inputs
        # Get target indices
        target_indices = np.random.randint(self.D + 1, size=self.batch_size)
        target_images = np.zeros((Agent.batch_size, 2048))

        for i, ti in enumerate(target_indices):
            target_images[i] = images[ti][i]

        fd = {}
        self.sender.fill_feed_dict(fd, target_images)
        self.receiver.fill_feed_dict(fd, images, target_indices)

        accuracy, loss, prediction = self.run_game(fd, mode=mode)

        return accuracy, loss
