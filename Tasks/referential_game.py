import numpy as np
from comet_ml import Experiment
import matplotlib.pyplot as plt
from utils.vocabulary import Vocabulary as V



from Agents import Agent


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
    def __init__(self, Sender, Receiver, data_handler=None, use_images=True, track_results=True, experiment=None):
        """
        :param K [Int]: Vocabulary Size
        :param D [Int]: Distractor Set Size
        :param use_images[Bool]: Whether to use images or one hot encoding (for debugging)
        """
        self.sender = Sender
        self.receiver = Receiver

        self.use_images = use_images

        self.dh = data_handler

        # Get necessary parameters from Agent
        self.batch_size = Agent.batch_size
        self.K = Agent.K  # Vocabulary Size
        self.D = Agent.D  # Distractor Set Size

        # Set up vocabulary
        self.V = V()
        try:
            self.V.load_counter()
        except FileNotFoundError:
            self.V.generate_vocab()
            self.V.save_counter()

        self.V.generate_top_k(self.K)

        # Get necessary ops to run
        self.run_ops = list(self.receiver.get_output()) + [Agent.step]
        self.train_ops = self.receiver.get_train_ops()

        # Set up comet tracking
        self.track_results = track_results
        self.train_metrics = {}
        self.val_metrics = {}
        self.experiment = experiment or Experiment(api_key='1jl4lQOnJsVdZR6oekS6WO5FI',
                                                   project_name='Referential Game',
                                                   auto_param_logging=False, auto_metric_logging=False,
                                                   disabled=(not track_results))
        self.params = {}
        self.params.update(Agent.get_params())
        self.params.update(self.dh.get_params())
        self.experiment.log_parameters(self.params)

    def get_experiment_key(self):
        return self.experiment.get_key()

    def train_epoch(self, e, mode="train"):
        """
        Play an epoch of a game defined by iterating over of each image of the dataset once (within a margin)
        For not using images, this is identical of play_game
        :return:
        """
        if not self.use_images:
            return self.train_batch(mode=mode)

        losses = []
        accuracies = []

        image_gen = self.dh.get_images(mode=mode)
        while True:
            try:
                images = next(image_gen)
                acc, loss = self.train_batch(images=images, mode=mode)
                losses.append(loss)
                accuracies.append(acc)
            except StopIteration:
                break

        avg_acc, avg_loss  = np.mean(accuracies), np.mean(losses)
        if mode == "val" and self.track_results:
            self.experiment.set_step(e)
            self.val_metrics["Validation Accuracy"] = avg_acc
            self.val_metrics["Validation Loss"] = avg_loss
            self.experiment.log_metrics(self.val_metrics)

        self.sender.save_model(self.experiment.get_key())
        self.receiver.save_model(self.experiment.get_key())

        return avg_acc, avg_loss

    def train_batch(self, images=None, captions=None, mode="train"):
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

        accuracy, loss, prediction = self.run_game(fd, mode=mode)

        if mode == "val":
            print(self.V.ids_to_tokens(prediction.T[0]))
            img = images[0][0]
            plt.axis('off')
            plt.imshow(img)
            plt.show()


        return accuracy, loss

    def run_game(self, fd, mode="train"):
        ops = self.run_ops + [self.sender.prediction]
        if mode == "train":
            ops += self.train_ops
        elif mode == "sender_train":
            ops += [self.train_ops[0]]
        elif mode == "receiver_train":
            ops += [self.train_ops[1]]
        acc, loss, step, prediction = Agent.sess.run(ops, feed_dict=fd)[:4]

        if mode[-5:] == "train" and self.track_results:
            stylized_mode = ' '.join(x.capitalize() for x in mode.split('_'))
            self.experiment.set_step(step)
            self.train_metrics[stylized_mode + "ing Accuracy"] = acc
            self.train_metrics[stylized_mode + "ing Loss"] = loss
            self.experiment.log_metrics(self.train_metrics)

        return acc, loss, prediction


