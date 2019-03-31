import string

import matplotlib.pyplot as plt
import numpy as np
from comet_ml import Experiment

from Agents.agent import Agent


class ImageCaptioning:
    """
    Class for running the image captioner
    """
    def __init__(self, image_captioner, data_handler=None, track_results=True, experiment=None):
        """ 
        Initialize this Image Captioning task
        """
        self.image_captioner = image_captioner

        # get params for agent
        self.L = Agent.L
        self.K = Agent.K
        self.batch_size = Agent.batch_size

        self.dh = data_handler

        # Get necessary ops to run
        self.run_ops = list(self.image_captioner.get_output()) + [Agent.step]
        self.train_ops = self.image_captioner.get_train_ops()

        self.V = Agent.V

        self.track_results = track_results
        self.train_metrics = {}
        self.val_metrics = {}
        self.experiment = experiment or Experiment(api_key='1jl4lQOnJsVdZR6oekS6WO5FI',
                                                   project_name='Image Captioning',
                                                   auto_param_logging=False, auto_metric_logging=False,
                                                   disabled=(not track_results))
        if experiment is None:
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
        losses = []
        accuracies = []

        image_gen = self.dh.get_images(return_captions=True, mode=mode)
        while True:
            try:
                images, captions = next(image_gen)
                acc, loss = self.train_batch(images[0], captions, mode=mode)
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

        return avg_acc, avg_loss

    def train_batch(self, images, captions, mode="train"):
        """
        Run the Image captioning to learn parameters
        """
        fd = {}

        in_captions, out_captions, loss_weights = self.get_useable_captions(captions)

        images = np.squeeze(images)

        self.image_captioner.fill_feed_dict(fd, images, in_captions, out_captions, loss_weights)
        accuracy, loss, prediction = self.run_game(fd, mode=mode)

        if mode == "val":
            print(self.V.ids_to_tokens(prediction[0]))
            print(self.V.ids_to_tokens(out_captions[0]))
            print(self.V.ids_to_tokens(in_captions[0]))
            print("----------------------------------")
            # img = images[0]
            # plt.axis('off')
            # plt.imshow(img)
            # plt.show()

        return accuracy, loss
        
    def get_useable_captions(self, captions):

        in_captions = np.zeros((self.image_captioner.batch_size, self.L))
        out_captions = np.zeros((self.image_captioner.batch_size, self.L))
        loss_weights = np.zeros((self.image_captioner.batch_size, self.L))

        for i, caption in enumerate(captions):
            # Randomly select caption to use
            chosen_caption = caption[0][np.random.randint(5)]
            tokens = chosen_caption.translate(str.maketrans('', '', string.punctuation))
            tokens = tokens.lower().split()
            loss_weights[i] = self.V.get_weights(self.L, tokens)
            out = self.V.tokens_to_ids(self.L, tokens)
            in_ = np.roll(out, 1)
            in_[0] = self.V.sos_id
            out_captions[i] = out
            in_captions[i] = in_

        return in_captions, out_captions, loss_weights

    def run_game(self, fd, mode="train"):
        ops = self.run_ops
        if mode == "train":
            ops += self.train_ops
        acc, loss, prediction, step = Agent.sess.run(ops, feed_dict=fd)[:4]

        if mode == "train" and self.track_results:
            self.experiment.set_step(step)
            self.train_metrics["Image Captioning Training Accuracy"] = acc
            self.train_metrics["Image Captioning Training Loss"] = loss
            self.experiment.log_metrics(self.train_metrics)

        return acc, loss, prediction
    