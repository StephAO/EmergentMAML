from Agents.receiver_agent import ReceiverAgent
from Agents.sender_agent import SenderAgent
import data_handler as dh
import numpy as np


class Referential_Game:
    """
    Class to play referential game, where a sender sees a target image and must send a message to a receiver, who must
    the pick the target image from a set of candidate images.
    Attributes:
        Sender [Agent]
        Receiver [Agent]
        Vocabulary Size [Int]
        Distractor Set Size [Int]
    """

    def __init__(self, K=500, D=500):
        """

        :param sender: sender in game
        :param receiver: receiver in game
        :param K [Int]: Vocabulary Size
        :param D [Int]: Distractor Set Size
        """
        self.sender = SenderAgent(K, D)
        recv_msg, hum_msg, msg_len = self.sender.get_output()
        self.receiver = ReceiverAgent(K, D, recv_msg, hum_msg, msg_len)
        # self.dh = dh.Data_Handler()
        self.batch_size = self.sender.batch_size
        self.K = K # Vocabulary Size
        self.D = D # Distractor Set Size


    def play_game(self):
        """
        Play a single instance of the game
        :return: None
        """
        # images = self.dh.get_images(imgs_per_batch=self.D + 1, num_batches=self.batch_size)
        target_indices = np.random.randint(self.D + 1, size=self.batch_size)

        # target_images = np.zeros(self.sender.batch_shape)
        # for i, ti in enumerate(target_indices):
        #     target_images[i] = images[ti][i]

        fd = {}
        self.sender.fill_feed_dict(fd, target_indices)
        self.receiver.fill_feed_dict(fd, [], target_indices)

        message, prediction, loss, e = self.receiver.run_game(fd)
        # print(message.shape, prediction)
        # print("-",target_indices[:3])
        # print("+",message[:3])

        accuracy = np.sum(prediction==target_indices) / np.float(self.batch_size)

        return loss, accuracy