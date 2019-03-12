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
        self.vocabulary = {}  # tok->id
        self.reverse_vocabulary = {}  # id -> tok
        self.vocabulary_counter = {}  # tok -> number of times it appears in all captions

        # initialize COCO caption handler
        self.coco_capts = COCO(self.coco_caption_file)

        # Add the start of sentence, end of sentence, and unknown tokens
        self.vocabulary[self.sos] = self.sos_id
        self.vocabulary[self.eos] = self.eos_id
        self.vocabulary[self.unk] = self.unk_id
        self.reverse_vocabulary[self.sos_id] = self.sos
        self.reverse_vocabulary[self.eos_id] = self.eos
        self.reverse_vocabulary[self.unk_id] = self.unk
        # Make sure they cannot get removed from dict
        self.vocabulary_counter[self.sos] = 1000000
        self.vocabulary_counter[self.eos] = 1000000
        self.vocabulary_counter[self.unk] = 1000000

        all_anns_ids = self.coco_capts.getAnnIds()
        id = 3
        lens = []
        total_ann = len(all_anns_ids)
        for ann_id in all_anns_ids:

            ann = self.coco_capts.loadAnns(ann_id)[0]
            tokens = ann['caption'].translate(str.maketrans('', '', string.punctuation))
            tokens = tokens.lower().split()
            lens.append(len(tokens))
            for tok in tokens:
                if tok == '':
                    continue
                if tok in self.vocabulary:
                    self.vocabulary_counter[tok] += 1
                else:
                    self.vocabulary[tok] = id
                    self.vocabulary_counter[tok] = 1
                    self.reverse_vocabulary[id] = tok
                    id += 1

        print("Maximum length: {}, average length: {}, std: {}".format(max(lens), np.mean(lens), np.std(lens)))
        assert id == len(self.vocabulary)
        assert len(list(self.vocabulary.values())) == len(set(self.vocabulary.values()))

    def save_vocab(self):
        with open(self.vocab_dir + 'vocabulary.p', 'wb+') as v, \
             open(self.vocab_dir + 'reverse_vocabulary.p', 'wb+') as rv, \
             open(self.vocab_dir + 'vocabulary_counter.p', 'wb+') as vc:
            pickle.dump(self.vocabulary, v)
            pickle.dump(self.reverse_vocabulary, rv)
            pickle.dump(self.vocabulary_counter, vc)

    def load_vocab(self):
        with open(self.vocab_dir + 'vocabulary.p', 'rb') as v, \
             open(self.vocab_dir + 'reverse_vocabulary.p', 'rb') as rv, \
             open(self.vocab_dir + 'vocabulary_counter.p', 'rb') as vc:
            self.vocabulary = pickle.load(v)
            self.reverse_vocabulary = pickle.load(rv)
            self.vocabulary_counter = pickle.load(vc)

    def get_top_k(self, k):
        new_vocab = {}
        new_rev_vocab = {}
        c = Counter(self.vocabulary_counter)
        for tok, _ in c.most_common(k):
            id = self.vocabulary[tok]
            new_vocab[tok] = id
            new_rev_vocab[id] = tok
        return new_vocab, new_rev_vocab

if __name__ == "__main__":
    v = Vocabulary()
    v.generate_vocab()
    v.save_vocab()
    # v.load_vocab()
    nv, nrv = v.get_top_k(20)
    print(nv)
    print(nrv)