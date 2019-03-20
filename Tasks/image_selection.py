import string

import matplotlib.pyplot as plt
import numpy as np
from comet_ml import Experiment

from Agents.agent import Agent
from Agents.image_selector import ImageSelector

from utils import data_handler as dh
from utils.vocabulary import Vocabulary as V

class ImageSelection:
    """
    class for running the image selector
    """
    
    def __init__(self, image_selector, data_handler=None, track_results=True, experiment=None):
        """
        Initialize the image selection task
        """
        self.L = Agent.L
        self.K = Agent.K
        self.D = Agent.D
        self.batch_size = Agent.batch_size

        self.image_selector = image_selector
        self.dh = data_handler

        # Get necessary ops to run
        self.run_ops = list(self.image_selector.get_output()) + [Agent.step]
        self.train_ops = self.image_selector.get_train_ops()
        
        self.V = V()
        try:
            self.V.load_vocab()
        except FileNotFoundError:
            self.V.generate_vocab()
            self.V.save_vocab()
            
        self.V.generate_top_k(self.K)

        # Setup comet experiment
        self.track_results = track_results
        self.train_metrics = {}
        self.val_metrics = {}
        self.experiment = experiment or Experiment(api_key='1jl4lQOnJsVdZR6oekS6WO5FI',
                                                   project_name='Image Selection',
                                                   auto_param_logging=False, auto_metric_logging=False,
                                                   disabled=(not track_results))
        self.experiment.log_multiple_params(Agent.get_params())


    def get_experiment_key(self):
        return self.experiment.get_key()
        
    def train_epoch(self, e, mode="train"):
        """
        :return:
        """
        losses = []
        accuracies = []

        image_gen = self.dh.get_images(return_captions=True, mode=mode)
        while True:
            try:
                images, captions = next(image_gen)
                acc, loss = self.train_batch(images, captions, mode=mode)
                losses.append(loss)
                accuracies.append(acc)
            except StopIteration:
                break

        avg_acc, avg_loss  = np.mean(accuracies), np.mean(losses)
        if mode == "val":
            self.experiment.set_step(e)
            self.val_metrics["Validation Accuracy"] = avg_acc
            self.val_metrics["Validation Loss"] = avg_loss
            self.experiment.log_multiple_metrics(self.val_metrics)

        return avg_acc, avg_loss
        
    def train_batch(self, images, captions, mode="train"):
        
        target_indices = np.random.randint(self.D + 1, size=self.batch_size)
        target_captions = np.zeros((self.L, self.batch_size, self.K))
        
        candidates = images
        
        for i, ti in enumerate(target_indices):
            chosen_caption = captions[i][ti][np.random.randint(5)]
            tokens = chosen_caption.translate(str.maketrans('', '', string.punctuation))
            tokens = tokens.lower().split()
            target_caption_ids = self.V.tokens_to_ids(self.L, tokens)
            target_captions_one_hot = np.zeros((self.L, self.K))
            target_captions_one_hot[np.arange(self.L),target_caption_ids] = 1
            target_captions[:,i] = target_captions_one_hot
        
        fd = {}
        self.image_selector.fill_feed_dict(fd, target_captions, candidates, target_indices)
        
        accuracy, loss = self.run_game(fd, mode=mode)

        return accuracy, loss
        
    def run_game(self, fd, mode="train"):
        ops = self.run_ops
        if mode == "train":
            ops += self.train_ops
        acc, loss, step = Agent.sess.run(ops, feed_dict=fd)[:3]

        if mode == "train" and self.track_results:
            self.experiment.set_step(step)
            self.train_metrics["Image Selection Training Accuracy"] = acc
            self.train_metrics["Image Selection Training Loss"] = loss
            self.experiment.log_multiple_metrics(self.train_metrics)

        return acc, loss