import string

import matplotlib.pyplot as plt
import numpy as np
from comet_ml import Experiment

from Agents.agent import Agent
from utils.vocabulary import Vocabulary as V


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

        # Set up vocabulary
        self.V = V()
        try:
            self.V.load_vocab()
        except FileNotFoundError:
            self.V.generate_vocab()
            self.V.save_vocab()
        self.vocabulary, self.reverse_vocabulary = self.V.get_top_k(self.K)
        
        # Set up comet tracking
        self.track_results = track_results
        self.train_metrics = {}
        self.val_metrics = {}
        self.experiment = experiment or Experiment(api_key='1jl4lQOnJsVdZR6oekS6WO5FI',
                                                   project_name='Reptile',
                                                   auto_param_logging=False, auto_metric_logging=False,
                                                   disabled=(not track_results))

        self.experiment.log_multiple_params(Agent.get_params())

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
            self.experiment.log_multiple_metrics(self.val_metrics)

        return avg_acc, avg_loss

    def train_batch(self, images, captions, mode="train"):
        """
        Run the Image captioning to learn parameters
        """
        fd = {}

        captions = self.get_useable_captions(captions)
        images = np.squeeze(images)

        self.image_captioner.fill_feed_dict(fd, images, captions)
        accuracy, loss, prediction = self.run_game(fd, mode=mode)

        if mode == "val" and False:
            print(self.ids_to_tokens(prediction.T[0]))
            img = images[0]
            plt.axis('off')
            plt.imshow(img)
            plt.show()

        return accuracy, loss

    def get_id(self, token):
        """
        Return the id given the corresponding token
        """
        return self.vocabulary.get(token, self.V.unk_id)

    def get_token(self, id):
        """
        Return the token given the corresponding id
        """
        return self.reverse_vocabulary.get(id, self.V.unk)

    def get_useable_captions(self, in_captions):
        captions = np.zeros((self.image_captioner.batch_size, self.L))

        for i, caption in enumerate(in_captions):
            # Randomly select caption to use
            chosen_caption = caption[np.random.randint(5)]
            tokens = chosen_caption.translate(str.maketrans('', '', string.punctuation))
            tokens = tokens.lower().split()
            captions[i] = self.tokens_to_ids(tokens)

            return captions

    def tokens_to_ids(self, tokens):
        """
        Map a sequence of tokens to ids
        """
        # Pad with eos if too short
        ids = np.full((self.L), self.V.eos_id)

        for j, tok in enumerate(tokens):
            # Truncate captions if too long (leave at least one eos tokens)
            if j >= self.L - 1:
                break
            ids[j] = self.get_id(tok)
        return ids

    def ids_to_tokens(self, ids):
        """
        Map a sequence of ids to tokens
        """
        # Pad with eos if too short
        tokens = []
        for id in ids:
            tokens.append(self.get_token(id))
        return ' '.join(tokens)

    def run_game(self, fd, mode="train"):
        ops = self.run_ops
        if mode == "train":
            ops += self.train_ops
        acc, loss, prediction, step = Agent.sess.run(ops, feed_dict=fd)[:4]

        if mode == "train" and self.track_results:
            self.experiment.set_step(step)
            self.train_metrics["Image Captioning Training Accuracy"] = acc
            self.train_metrics["Image Captioning Training Loss"] = loss
            self.experiment.log_multiple_metrics(self.train_metrics)

        return acc, loss, prediction