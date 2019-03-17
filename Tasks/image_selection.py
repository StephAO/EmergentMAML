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
    
    def __init__(self, K, D, L, loss_type='pairwise', track_results=True):
        """
        Initialize the image selection task
        """
        self.L = L
        self.K = K
        self.D = D 

        self.dh = dh.Data_Handler()
        
        self.V = V()
        try:
            self.V.load_vocab()
        except FileNotFoundError:
            self.V.generate_vocab()
            self.V.save_vocab()
            
        self.V.generate_top_k(K)
        
        self.image_selector = ImageSelector(K, D, L, use_images=True)
        self.batch_size = Agent.batch_size
        
        self.train_metrics = {}
        self.val_metrics = {}
        self.experiment = Experiment(api_key='1jl4lQOnJsVdZR6oekS6WO5FI',
                                     project_name='Emergent_MAML',
                                     auto_param_logging=False, auto_metric_logging=False,
                                     disabled=(not track_results))

        self.experiment.log_multiple_params(Agent.get_params())
        
    def train_epoch(self, e, data_type="train"):
        """
        :return:
        """
        losses = []
        accuracies = []

        image_gen = self.dh.get_images(images_per_instance=self.D + 1, 
            batch_size=self.batch_size, return_captions=True,
            data_type=data_type)
        while True:
            try:
                images, captions = next(image_gen)
                acc, loss = self.train_batch(images, captions, data_type=data_type)
                losses.append(loss)
                accuracies.append(acc)
            except StopIteration:
                break

        avg_acc, avg_loss  = np.mean(accuracies), np.mean(losses)
        if data_type == "val":
            self.experiment.set_step(e)
            self.val_metrics["Validation Accuracy"] = avg_acc
            self.val_metrics["Validation Loss"] = avg_loss
            self.experiment.log_multiple_metrics(self.val_metrics)

        return avg_acc, avg_loss
        
    def train_batch(self, images, captions, data_type="train"):
        
        target_indices = np.random.randint(self.D + 1, size=self.batch_size)
        target_captions = np.zeros((self.L, self.batch_size, self.K))
        
        candidates = images
        
        for i, ti in enumerate(target_indices):
            chosen_caption = captions[i][target_indices[i]][np.random.randint(5)]
            tokens = chosen_caption.translate(str.maketrans('', '', string.punctuation))
            tokens = tokens.lower().split()
            target_caption_ids = self.V.tokens_to_ids(self.L, tokens)
            target_captions_one_hot = np.zeros((self.L, self.K))
            target_captions_one_hot[np.arange(self.L),target_caption_ids] = 1
            target_captions[:,i] = target_captions_one_hot
        
        fd = {}
        self.image_selector.fill_feed_dict(fd, target_captions, candidates, target_indices)
        
        accuracy, loss, prediction, step = self.image_selector.run_game(fd, data_type=data_type)

        if data_type == "train":
            self.experiment.set_step(step)
            self.train_metrics["Training Accuracy"] = accuracy
            self.train_metrics["Training Loss"] = loss
            self.experiment.log_multiple_metrics(self.train_metrics)

        return accuracy, loss
        
        