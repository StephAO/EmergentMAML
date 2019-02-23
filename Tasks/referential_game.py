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

    def __init__(self, K=10, D=1, use_images=True):
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

        #### Just debugging
        if (e-1) % 5 == 0:# and self.use_images:
            # f, f_, p, e_ = self.receiver.sess.run([self.receiver.image_features, self.receiver.rnn_features, self.receiver.img_feat_1, self.receiver.energy_tensor], feed_dict=fd)
            sender, receiver = self.receiver.sess.run([self.sender.pre_feat, self.receiver.img_feat_1], feed_dict=fd)

            # before, mid, after = np.array(before), np.array(mid), np.array(after)

            # fig = plt.figure()
            # plt.axis('off')
            # fig.patch.set_facecolor('xkcd:gray')
            # fig.add_subplot(3, 2, 1)
            # plt.imshow(before[0][0])
            #
            # fig.add_subplot(3, 2, 2)
            # plt.imshow(before[1][0])
            #
            # fig.add_subplot(3, 2, 3)
            # plt.imshow(mid[0][:, :64])
            #
            # fig.add_subplot(3, 2, 4)
            # plt.imshow(mid[1][:, :64])
            #
            # fig.add_subplot(3, 2, 5)
            # plt.imshow(after[0])
            #
            # fig.add_subplot(3, 2, 6)
            # plt.imshow(after[1])

            # print(mid.shape)
            # print(np.all(np.equal(mid[0], mid[1])))
            # print(np.mean(sender), np.mean(receiver))
            # print(sender[0], sender[1], receiver[0][0], receiver[1][0])
            # print("--------")
            # print(c[95:97, :32].shape, mid[:, 0, :32].shape, after[:, 0, :32].shape)
            # plt.show()


            # print(message[0][0][:4])

            # plt.axis('off')
            # print(np.shape(p))
            # a = np.concatenate(p, axis=0)
            # print(np.shape(a))
            # plt.imshow(a)
            #
            # plt.show()

            # for i in p[:4 ]:
            #     print(i[0][:4])

            # for i in f[:4 ]:
            #     print(i[0][:4])
            #
            # print(f_[0][:4])
            #
            # print(e_[0])

            # energies, target_energy, rnn = self.receiver.sess.run(
            #     [self.receiver.energy_tensor, self.receiver.target_energy, self.receiver.rnn_features], feed_dict=fd)
            #
            # print("==>", np.mean(np.abs(np.expand_dims(target_energy, axis=1) - energies)), np.max(energies))
        # print(target_energy[0])
        # print(target_indices)

        #####
        # print(self.receiver.sess.run([self.receiver.diff_loss], feed_dict=fd))

        message, prediction, loss = self.receiver.run_game(fd)
        # print("==>", message[0][0], target_indices[0])

        accuracy = np.sum(prediction==target_indices) / np.float(self.batch_size)

        return loss, accuracy