import string

import matplotlib.pyplot as plt
import numpy as np
from comet_ml import Experiment

from Agents.agent import Agent
from Tasks import Task


class ImageCaptioning(Task):
    """
    Class for running the image captioner
    """
    def __init__(self, *args, **kwargs):
        self.name = "Image Captioning"
        super().__init__(*args, **kwargs)
        self.captions_required = True

    def train_batch(self, inputs, mode="train"):
        """
        Run the Image captioning to learn parameters
        """
        images, captions = inputs
        images = np.squeeze(images)

        in_captions, out_captions = self.get_useable_captions(captions)

        fd = {}
        self.agents[0].fill_feed_dict(fd, images, in_captions, out_captions)
        accuracy, loss, prediction = self.run_game(fd, mode=mode)

        return accuracy, loss

    def get_useable_captions(self, captions):
        """
        returns captions that can be fed to an image captioner (sequences of ids)
        in_captions are prepended by a sos token, while out_captions are the full sequence.
        Both are capped by the maximum length of a caption
        :param captions:
        :return:
        """
        in_captions = np.zeros((self.batch_size, self.L))
        out_captions = np.zeros((self.batch_size, self.L))

        for i, caption in enumerate(captions):
            # Randomly select caption to use
            chosen_caption = caption[0][np.random.randint(5)]
            tokens = chosen_caption.translate(str.maketrans('', '', string.punctuation))
            tokens = tokens.lower().split()
            out_captions[i] = self.V.tokens_to_ids(self.L, tokens)
            in_captions[i] = np.roll(out_captions[i], 1)
            in_captions[i][0] = self.V.sos_id

        return in_captions, out_captions
    