import matplotlib.pyplot as plt
import numpy as np

import data_handler as dh
from Agents.image_captioner import ImageCaptioner
import string
from utils.vocabulary import Vocabulary as V


class ImageCaptioning:
    """
    Class for running the image captioner
    """
    def __init__(self, K=500, L=15):
        """ 
        Initialize this Image Captioning task
        """
        self.L = L
        self.K = K

        
        self.dh = dh.Data_Handler()
        # TODO - (1) need to get annotations to construct vocabulary --> find a better way
        # TODO - (2) Since I don't know in advance which images will be used for all the batches, I construct the vocab using a subset of images
        
        self.V = V()
        self.V.load_vocab()
        self.vocabulary, self.reverse_vocabulary = self.V.get_top_k(K)
        
        self.image_captioner = ImageCaptioner(self.K, 0, use_images=True)
        self.batch_size = self.image_captioner.batch_size
        
    def train(self):
        """
        Run the Image captioing to learn parameters
        """
        fd = {}
        
        imgs, captions = self.dh.get_images(
            imgs_per_batch=1, num_batches=self.batch_size, return_captions=True)

        imgs = imgs[0]
            
        captions = self.get_caption_vector(captions)

        self.image_captioner.fill_feed_dict(fd, imgs, captions)
        accuracy, loss = self.image_captioner.run_game(fd)
        #
        # # TODO - why did prediction have the shape [caption_len, batch_size] and not the other way around?
        # accuracy = np.divide(np.sum(np.equal(prediction.T, captions)), np.float(self.batch_size))
        #
        return loss, accuracy

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

    def get_caption_vector(self, in_captions):
        """
        Get the vector associated with the caption of maximum length L
        L must be at least 3 (eos and sos and 1 token)
        """
        # Pad with eos if too short
        captions = np.full((self.image_captioner.batch_size, self.L), self.V.eos_id)

        for i, caption in enumerate(in_captions):
            chosen_caption = caption[np.random.randint(5)]
            tokens = chosen_caption.translate(str.maketrans('', '', string.punctuation))
            tokens = tokens.lower().split()
            for j, tok in enumerate(tokens):
                # Truncate captions if too long
                if j >= self.L - 1:
                    break
                captions[i, j] = self.get_id(tok)

        return captions
        
if __name__ == "__main__":
    # For testing the image captioner on its own
    epochs = 10000
    
    ic = ImageCaptioning()
    
    losses = []
    accuracies = []
    for e in range(1, epochs + 1):
        loss, acc = ic.train()

        # Print and collect stats
        if (e) % 20 == 0:
            print("loss: {0:1.4f}, accuracy: {1:3.2f}%".format(np.mean(losses[-20:]), np.mean(accuracies[-20:])*100))
        if e % 100 == 0:
            print("--- EPOCH {0:5d} ---".format(e))
        losses.append(loss)
        accuracies.append(acc)

        # 100% Success - end training
        if np.mean(accuracies[-20:]) == 1.0:
            break

    print("--- EPOCH {0:5d} ---".format(e))
    ml = max(losses)
    losses_ = [l / ml for l in losses]
    accuracies_ = [np.mean(accuracies[i: i + 10]) for i in range(len(accuracies) - 10)]
    
    plt.plot(losses_, 'r', accuracies_, 'g')  # , lrs, 'b')
    # plt.show()
    plt.savefig("./output_graph.png")
    
    
    
    
    
    
    
    
    
    
    