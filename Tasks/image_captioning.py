from Agents.image_captioner import ImageCaptioner
import data_handler as dh
import numpy as np

# TODO - Move to another file/Merge with ImageCaptioing
class Vocabulary:
    
    def __init__(self, annons, vocab_size=1000):
        """
        Construct the vocabulary used in image captioing task
        """
        # special tokens
        self.sos = '<s>'
        self.eos = '</s>'
        self.unk = '<unk>'
        
        self.sos_id = 0
        self.eos_id = 1
        self.unk_id = 2
        
        self.token_to_id = {} # maps a token to an int
        
        # Add the start of sentence, end of sentence and unknown tokens
        self.token_to_id[self.sos] = self.sos_id
        self.token_to_id[self.eos] = self.eos_id
        self.token_to_id[self.unk] = self.unk_id
        
        self.vocab_size = 3 # at most vocab_size tokens
        
        batch_idx = 0
        img_idx = 0
        curr_tokens = None
        tokens_idx = 0
        
        # [Q] some notes that we might need to change
        # 1. I only consider the first annotation for each image when constructing
        #    the vocabulary 
        # 2. I keep parsing until I reach the threshold, a better approach would be to randomly include a 
        #    subset of the sparsely occuring tokens?
        
        while (self.vocab_size < vocab_size):
            if batch_idx >= len(annons):
                break
            
            if img_idx >= len(annons[batch_idx]):
                batch_idx += 1
                img_idx = 0
                curr_tokens = None
            else:
                if curr_tokens == None:
                    # TODO - currently, I just split on whitespaces 
                    curr_tokens = \
                        annons[batch_idx][img_idx][0]['caption'].lower().split()
                    tokens_idx = 0
                    
                if tokens_idx >= len(curr_tokens):
                    img_idx += 1 
                    curr_tokens = None
                    tokens_idx = 0
                else:
                    if curr_tokens[tokens_idx] not in self.token_to_id:
                        self.token_to_id[curr_tokens[tokens_idx]] = \
                            self.vocab_size 
                        self.vocab_size += 1
                    tokens_idx += 1
        
        self.tokens = sorted(self.token_to_id, key=self.token_to_id.get)
        
    def get_id(self, token):
        """
        Return the corresponding id associated with the token
        """
        token_lower = token.lower()
        if token_lower in self.token_to_id:
            return self.token_to_id[token_lower]
        else:
            return self.unk_id
            
    def get_token(self, id):
        """
        Return the token given the corresponding
        """
        if id >= len(self.tokens):
            return self.unk
        return self.tokens[id]
    
    def get_caption_vector(self, caption, L=10):
        """
        Get the vector associated with the caption of maximum length L
        L must be at least 3 (eos and sos and 1 token)
        """
        assert L >= 3
        
        res = np.zeros(L)
        
        # [Q]
        # - what if the caption has shorter len - for now, add more eos tokens
        # - what if the caption has longer len - for now, truncate
        res[0] = self.sos_id
        res[(L - 1)] = self.eos_id 
        
        # TODO - I just set to lowercase and split on white space - might need to do more processing
        tokens = caption.lower().split()
        t = 1
        while t < (L - 1):
            res[t] = self.get_id(tokens[t - 1])
            t += 1
            
        # TODO - may be instead of returning the vector, accept and modify the matrix
        return res
        
class ImageCaptioning:
    """
    Class for running the image captioner
    """
    
    captions_num_for_vocab = 100 # find a better way for constructing the vocab
    
    def __init__(self, batch_size=16, caption_len=10):
        """ 
        Initialize this IamageCaptioing task
        """
        self.batch_size = batch_size
        
        # [Q] does L include the sos and eos tokens? for now, no
        # TODO - for now, caption length doesn't include the sos and eos tokens
        self.L = caption_len + 2
        
        self.dh = dh.Data_Handler()
        # TODO - (1) need to get annotations to construct vocabulary --> find a better way
        # TODO - (2) Since I don't know in advance which images will be used for all the batches, I construct the vocab using a subset of images
        # [Q] would it make sense to change the vocab for each batch? for now, no 
        _, all_captions = self.dh.get_images(
            imgs_per_batch=ImageCaptioning.captions_num_for_vocab, num_batches=1, captions=True)
        
        self.vocabulary = Vocabulary(all_captions, vocab_size=100)
        
        self.image_captioner = ImageCaptioner(self.vocabulary.vocab_size, L=self.L)
        
    def train(self):
        """
        Run the Image captioing to learn parameters
        """
        fd = {}
        
        imgs, annons = self.dh.get_images(
            imgs_per_batch=1, num_batches=self.batch_size, captions=True)
            
        imgs = imgs[0]
        captions = np.zeros([self.batch_size, self.L], dtype=np.int32)
        
        for b in range(len(annons)):
            captions[b] = self.vocabulary.get_caption_vector(annons[b][0][0]['caption'], L=self.L)
        
        self.image_captioner.fill_feed_dict(fd, imgs, captions)
        
        prediction, loss = self.image_captioner.run_game(fd)
        
        # TODO - why did prediction have the shape [caption_len, batch_size] and not the other way around?
        accuracy = np.divide(np.sum(np.equal(prediction.T, captions)), np.float(self.batch_size))

        return loss, accuracy
        
if __name__ == "__main__":
    # For testing the image captioner on its own
    epochs = 10000
    
    ic = ImageCaptioning(batch_size=16, caption_len=1)
    
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
    
    
    
    
    
    
    
    
    
    
    