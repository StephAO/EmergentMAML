from Agents.receiver_agent import ReceiverAgent
from Agents.sender_agent import SenderAgent
import data_handler as dh
import matplotlib.pyplot as plt
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

    def __init__(self, K=2, D=1, use_images=True):
        """

        :param sender: sender in game
        :param receiver: receiver in game
        :param K [Int]: Vocabulary Size
        :param D [Int]: Distractor Set Size
        """
        self.use_images = use_images
        self.sender = SenderAgent(K, D, use_images=use_images)
        recv_msg, hum_msg, msg_len = self.sender.get_output()
        self.receiver = ReceiverAgent(K, D, recv_msg, hum_msg, msg_len, use_images=use_images)
        if use_images:
            self.dh = dh.Data_Handler()
        self.batch_size = self.sender.batch_size
        self.K = K # Vocabulary Size
        self.D = D # Distractor Set Size


    def play_game(self, e):
        """
        Play a single instance of the game
        :return: None
        """
        target_indices = np.random.randint(self.D + 1, size=self.batch_size)
        target_images = np.zeros(self.sender.batch_shape)

        target = target_indices
        candidates = []

        if self.use_images:
            images = self.dh.get_images(imgs_per_batch=self.D + 1, num_batches=self.batch_size)
            for i, ti in enumerate(target_indices):
                target_images[i] = images[ti][i]
            target = target_images
            candidates = images

            # print(np.shape(candidates))
            #
            # plt.imshow(images[0][0])
            # plt.show()
            # plt.imshow(images[1][0])
            # plt.show()

        fd = {}
        self.sender.fill_feed_dict(fd, target)
        self.receiver.fill_feed_dict(fd, candidates, target_indices)

        message, prediction, loss = self.receiver.run_game(fd)

        #### Just debugging
        if e % 100 == 0:
            f, f_, p, e_ = self.receiver.sess.run([self.receiver.image_features, self.receiver.rnn_features, self.receiver.img_feat_1, self.receiver.energy_tensor], feed_dict=fd)
            print(message[0][0][:4])

            # plt.axis('off')
            # print(np.shape(p))
            # a = np.concatenate(p, axis=0)
            # print(np.shape(a))
            # plt.imshow(a)
            #
            # plt.show()

            # for i in p[:4 ]:
            #     print(i[0][:4])

            for i in f[:4 ]:
                print(i[0][:4])

            print(f_[0][:4])

            print(e_[0])
        #####

        accuracy = np.sum(prediction==target_indices) / np.float(self.batch_size)

        return loss, accuracy