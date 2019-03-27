from collections import Counter
import numpy as np
import os
import pickle
from pycocotools.coco import COCO
import string
import sys

class Vocabulary:
    def __init__(self):
        """
        Construct the vocabulary used in image captioing task
        """
        # The actual vocabulary
        self.vocabulary = {}
        self.reverse_vocabulary = {}
        self.vocabulary_counter = {}

        # special tokens
        self.sos = '<s>'
        self.eos = '</s>'
        self.unk = '<unk>'
        self.sos_id = 0
        self.eos_id = 1
        self.unk_id = 2

        # MSCOCO handler
        self.coco_data_dir = '/home/stephane/cocoapi'
        self.coco_dataType = 'train2014'
        self.coco_caption_file = '{}/annotations/captions_{}.json'.format(self.coco_data_dir, self.coco_dataType)

        # Load/Save
        self.base_dir = '/home/stephane/EmergentMAML' #os.path.dirname(sys.modules['__main__'].__file__)
        self.data_dir = self.base_dir + '/data/'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def generate_vocab(self):
        """
        Generate vocabulary counter dictionary from MSCOCO dataset. Counts the occurences of each token found
        :return:
        """
        # The actual vocabulary
        self.vocabulary_counter = {}  # tok -> number of times it appears in all captions

        # initialize COCO caption handler
        self.coco_capts = COCO(self.coco_caption_file)

        all_anns_ids = self.coco_capts.getAnnIds()
        for ann_id in all_anns_ids:
            ann = self.coco_capts.loadAnns(ann_id)[0]
            tokens = ann['caption'].translate(str.maketrans('', '', string.punctuation))
            tokens = tokens.lower().split()
            for tok in tokens:
                if tok == '':
                    continue
                self.vocabulary_counter[tok] = self.vocabulary_counter.get(tok, 0) + 1

        print("Created vocabulary with {} different tokens".format(len(self.vocabulary_counter)))


    def save_vocab(self):
        with open(self.data_dir + 'vocabulary_counter.p', 'wb+') as vc:
            pickle.dump(self.vocabulary_counter, vc)

    def load_vocab(self):
        with open(self.data_dir + 'vocabulary_counter.p', 'rb') as vc:
            self.vocabulary_counter = pickle.load(vc)

    def generate_top_k(self, k):
        """
        Generates a dictionary with start of sentence, end of sentence, unknown tokens + (k-3) most common tokens from the
        generated dictionary (see generate_vocab)
        :param k:
        :return:
        """
        # Add the start of sentence, end of sentence, and unknown tokens
        self.vocabulary[self.sos] = self.sos_id
        self.vocabulary[self.eos] = self.eos_id
        self.vocabulary[self.unk] = self.unk_id
        self.reverse_vocabulary[self.sos_id] = self.sos
        self.reverse_vocabulary[self.eos_id] = self.eos
        self.reverse_vocabulary[self.unk_id] = self.unk

        c = Counter(self.vocabulary_counter)
        id = 3
        # create dictionary so that index
        for tok, _ in c.most_common(k - 3):
            self.vocabulary[tok] = id
            self.reverse_vocabulary[id] = tok
            id += 1
    
    def get_id(self, token):
        """
        Return the id given the corresponding token
        """
        return self.vocabulary.get(token, self.unk_id)
    
    def get_token(self, id):
        """
        Return the token given the corresponding id
        """
        return self.reverse_vocabulary.get(id, self.unk)
        
    def tokens_to_ids(self, L, tokens):
        """
        Map a sequence of tokens to ids
        """
        # Pad with eos if too short
        ids = np.full((L), self.eos_id)
        idx = 0

        for tok in tokens:
            # Truncate captions if too long (leave at least one eos tokens)
            if idx >= L:
                break
            # TODO not sure if this a good or bad thing
            # Avoid uninformative tokens
            # if tok in ["", "a", "an", "the"]:
            #     continue
            ids[idx] = self.get_id(tok)
            idx += 1

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
    
if __name__ == "__main__":
    v = Vocabulary()
    v.generate_vocab()
    # v.save_vocab()
    # v.load_vocab()
