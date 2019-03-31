import string

import matplotlib.pyplot as plt
import numpy as np
from comet_ml import Experiment

from Agents import Agent
from Tasks import Task

class ImageSelection(Task):
    """
    class for running the image selector
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Image Selection"
        self.captions_required = True
        
    def train_batch(self, inputs, mode="train"):
        images, captions = inputs

        target_indices = np.random.randint(self.D + 1, size=self.batch_size)
        target_captions = np.zeros((self.batch_size, self.L, self.K))
        
        candidates = images
        
        for i, ti in enumerate(target_indices):
            chosen_caption = captions[i][ti][np.random.randint(5)]
            # print(chosen_caption)
            tokens = chosen_caption.translate(str.maketrans('', '', string.punctuation))
            tokens = tokens.lower().split()
            target_caption_ids = self.V.tokens_to_ids(self.L, tokens)
            target_captions_one_hot = np.zeros((self.L, self.K))
            target_captions_one_hot[np.arange(self.L),target_caption_ids] = 1
            target_captions[i] = target_captions_one_hot
        
        fd = {}
        self.agents[0].fill_feed_dict(fd, target_captions, candidates, target_indices)
        
        accuracy, loss, prediction = self.run_game(fd, mode=mode)

        return accuracy, loss

    def visual_analysis(self, index, caption, images):
        fig = plt.figure(figsize=(10, 10))
        columns = 8
        rows = 4
        print(self.V.ids_to_tokens(caption))
        print(index)
        for i, img in enumerate(images):
            fig.add_subplot(rows, columns, i+1)
            plt.imshow(img[-1])
        plt.show()
