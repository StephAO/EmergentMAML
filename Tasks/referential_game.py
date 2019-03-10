from Agents.receiver_agent import ReceiverAgent
from Agents.sender_agent import SenderAgent
import data_handler as dh
import matplotlib.pyplot as plt
import numpy as np


class ReferentialGame:
    """
    Class to play referential game, where a sender sees a target image and must send a message to a receiver, who must
    the pick the target image from a set of candidate images.
    Attributes:
        Sender [Agent]
        Receiver [Agent]
        Vocabulary Size [Int]
        Distractor Set Size [Int]
    """

    def __init__(self, K=100, D=1, use_images=True, loss_type='pairwise'):
        """
        :param K [Int]: Vocabulary Size
        :param D [Int]: Distractor Set Size
        :param use_images[Bool]: Whether to use images or one hot encoding (for debugging)
        """
        self.use_images = use_images
        self.sender = SenderAgent(K, D, use_images=use_images)
        recv_msg, hum_msg, msg_len = self.sender.get_output()
        self.receiver = ReceiverAgent(K, D, recv_msg, msg_len, use_images=use_images, loss_type=loss_type)
        if use_images:
            self.dh = dh.Data_Handler()
        self.batch_size = self.sender.batch_size
        self.K = K  # Vocabulary Size
        self.D = D  # Distractor Set Size

    def play_epoch(self):
        """
        Play an epoch of a game defined by iterating over of each image of the dataset once (within a margin)
        For not using images, this is identical of play_game
        :return:
        """
        if not self.use_images:
            return self.play_game()

        image_gen = self.dh.get_images(imgs_per_batch=self.D + 1, num_batches=self.batch_size)
        while True:
            try:
                images = next(image_gen)
                message, accuracy, loss = self.play_game(images=images)
            except StopIteration:
                break

        return loss, accuracy

    def play_game(self, images=None):
        """
        Play a single instance of the game
        :return:
        """
        # Get target indices
        target_indices = np.random.randint(self.D + 1, size=self.batch_size)
        target_images = np.zeros(self.sender.batch_shape)

        target = target_indices
        candidates = []

        if self.use_images:
            for i, ti in enumerate(target_indices):
                target_images[i] = images[ti][i]
            target = target_images
            candidates = images

        # TODO can this be done using a tf data iterator?
        fd = {}
        self.sender.fill_feed_dict(fd, target)
        self.receiver.fill_feed_dict(fd, candidates, target_indices)

        message, accuracy, loss = self.receiver.run_game(fd)

        return loss, accuracy
