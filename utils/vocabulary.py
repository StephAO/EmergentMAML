from collections import Counter, defaultdict
import numpy as np
import os
import pickle
from pycocotools.coco import COCO
import string
import sys
from .data_handler import coco_path, project_path

class Vocabulary:
    def __init__(self, embedding_len):
        """
        Construct the vocabulary used in image captioing task
        """
        if embedding_len not in [50, 100, 200, 300]:
            raise ValueError("Embedding size must be one of [50, 100, 200, 300]")

        # The actual vocabulary
        self.tok_to_idx = {}
        self.idx_to_tok = {}
        self.idx_to_emb = []
        self.vocabulary_counter = {}
        self.embedding_len = embedding_len


        # special tokens
        self.sos = '<s>'
        self.eos = '</s>'
        self.unk = '<unk>'
        self.sos_id = 0
        self.eos_id = 1
        self.unk_id = 2
        self.sos_emb = np.full(self.embedding_len, 1)
        self.eos_emb = np.full(self.embedding_len, -1)
        self.unk_emb = np.zeros(self.embedding_len)

        # MSCOCO handler
        self.coco_data_dir = coco_path
        self.coco_dataType = 'train2014'
        self.coco_caption_file = '{}/annotations/captions_{}.json'.format(self.coco_data_dir, self.coco_dataType)

        # Load/Save
        self.base_dir = project_path
        self.data_dir = self.base_dir + '/data/'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # Glove embedding
        self.glove_emb_path = '/home/stephane/glove.840B.300d.txt'.format(self.embedding_len)

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


    def save_counter(self):
        with open(self.data_dir + 'vocabulary_counter.p', 'wb+') as vc:
            pickle.dump(self.vocabulary_counter, vc)

    def load_counter(self):
        with open(self.data_dir + 'vocabulary_counter.p', 'rb') as vc:
            self.vocabulary_counter = pickle.load(vc)

    def save_embedding(self, K):
        with open(self.data_dir + 'embedding_{}.p'.format(K), 'wb+') as e:
            pickle.dump((self.idx_to_emb, self.tok_to_idx, self.idx_to_tok), e)

    def load_embedding(self, K):
        with open(self.data_dir + 'embedding_{}.p'.format(K), 'rb') as e:
            self.idx_to_emb, self.tok_to_idx, self.idx_to_tok = pickle.load(e)

    def generate_top_k2(self, k):
        """
        Generates a dictionary with start of sentence, end of sentence, unknown tokens + (k-3) most common tokens from the
        generated dictionary (see generate_vocab)
        :param k:
        :return:
        """
        # Add the start of sentence, end of sentence, and unknown tokens
        self.tok_to_idx[self.sos] = self.sos_id
        self.tok_to_idx[self.eos] = self.eos_id
        self.tok_to_idx[self.unk] = self.unk_id
        self.idx_to_tok[self.sos_id] = self.sos
        self.idx_to_tok[self.eos_id] = self.eos
        self.idx_to_tok[self.unk_id] = self.unk

        c = Counter(self.vocabulary_counter)
        id = 3
        # create dictionary so that index
        for tok, _ in c.most_common(k - 3):
            self.tok_to_idx[tok] = id
            self.idx_to_tok[id] = tok
            id += 1

    def generate_top_k(self, k):
        """
        Generates a dictionary with start of sentence, end of sentence, unknown tokens + (k-3) most common tokens from the
        generated dictionary (see generate_vocab)
        :param k:
        :return:
        """
        try:
            self.load_embedding(k)
            return
        except FileNotFoundError:
            pass

        # Add the start of sentence, end of sentence, and unknown tokens
        self.idx_to_emb.append(self.sos_emb)
        self.idx_to_emb.append(self.eos_emb)
        self.idx_to_emb.append(self.unk_emb)

        self.tok_to_idx[self.sos] = self.sos_id
        self.tok_to_idx[self.eos] = self.eos_id
        self.tok_to_idx[self.unk] = self.unk_id

        c = Counter(self.vocabulary_counter)
        common_words = c.most_common(k - 3)
        relevant_words = [x[0] for x in common_words]
        self.load_embedding_from_disks(relevant_words)
        idx = len(self.tok_to_idx) + 3
        for rw in relevant_words:
            if rw not in self.tok_to_idx:
                print(rw, self.vocabulary_counter[rw])
                self.tok_to_idx[rw] = idx
                self.idx_to_emb.append(self.unk_emb)
                idx += 1

        assert(len(self.tok_to_idx) == k)

        # create inverse dictionary
        self.idx_to_tok = {v: k for k, v in self.tok_to_idx.items()}
        self.save_embedding(k)
    
    def get_id(self, token):
        """
        Return the id given the corresponding token
        """
        return self.tok_to_idx.get(token, self.unk_id)
    
    def get_token(self, id):
        """
        Return the token given the corresponding id
        """
        return self.idx_to_tok.get(id, self.unk)
        
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
            if tok in ["", "a", "an", "the"]:
                continue
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

    def ids_to_embs(self, ids):
        embedding = np.zeros((len(ids), self.embedding_len))
        for i, id in enumerate(ids):
            embedding[i] = self.idx_to_emb[id]
        return embedding

    def tokens_to_embs(self, tokens):
        embedding = np.zeros((len(tokens), self.embedding_len))
        for i, tok in enumerate(tokens):
            embedding[i] = self.idx_to_emb[self.tok_to_idx[tok]]
        return embedding

    def load_embedding_from_disks(self, relevant_words):
        """
        Read a GloVe txt file. Only return entries for list of relevant words
        If `with_indexes=True`, we return a tuple of two dictionaries
        `(tok_to_idx, idx_to_emb)`, otherwise we return only a direct
        `word_to_embedding_dict` dictionary mapping from a string to a numpy array.
        """
        idx = 3
        with open(self.glove_emb_path, 'r') as glove_file:
            for line in glove_file:
                split = line.split(' ')

                word = split[0]
                if word not in relevant_words:
                    continue

                representation = split[1:]
                representation = np.array(
                    [float(val) for val in representation]
                )

                self.tok_to_idx[word] = idx
                self.idx_to_emb.append(representation)
                idx += 1

if __name__ == "__main__":
    v = Vocabulary(300)
    v.generate_vocab()
    v.generate_top_k(10)
    print(v.tok_to_idx)
    v.generate_top_k2(10)
    print(v.tok_to_idx)
    # print(v.vocabulary_counter['frisbee'], v.vocabulary_counter['frisbe'])
    # print(v.token_to_idx)
    # v.save_counter()
    # v.load_counter()
