from collections import Counter
import numpy as np
import pickle
from pycocotools.coco import COCO
import string

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
        self.vocab_dir = '/home/stephane/PycharmProjects/EmergentMAML/data/'

    def generate_vocab(self):
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


    def save_vocab(self):
        with open(self.vocab_dir + 'vocabulary_counter.p', 'wb+') as vc:
            pickle.dump(self.vocabulary_counter, vc)

    def load_vocab(self):
        with open(self.vocab_dir + 'vocabulary_counter.p', 'rb') as vc:
            self.vocabulary_counter = pickle.load(vc)

    def get_top_k(self, k):
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

        return self.vocabulary, self.reverse_vocabulary

if __name__ == "__main__":
    v = Vocabulary()
    # v.generate_vocab()
    # v.save_vocab()
    v.load_vocab()
    nv, nrv = v.get_top_k(20)
    print(nv)
    print(nrv)