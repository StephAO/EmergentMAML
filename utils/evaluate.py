import numpy as np
import matplotlib.pyplot as plt
import os
from pycocotools.coco import COCO
import pickle
import string
import tensorflow as tf
from collections import Counter
import nltk
from tqdm import tqdm

# TODO: consider moving this to a better spot
from utils.data_handler import coco_path, project_path
from Agents import Agent, ImageCaptioner, ImageSelector
from utils.BLEU import *

class Evaluator:

    def __init__(self, image_captioner, image_selector):
        self.ic = image_captioner
        self.is_ = image_selector
        self.ops = [self.ic.prediction, self.ic.probabilities]
        self.batch_size = Agent.batch_size
        self.V = Agent.V
        self.L = Agent.L
        # Word counters
        self.generated_word_counter = {}
        self.caption_word_counter = {}
        self.omission_scores = []
        self.bleu_scores = {ngram: [] for ngram in [1,2,3,4] }
        
        # MSCOCO variables
        self.coco_path = coco_path
        self.feat_dir = 'train_feats'
        self.data_dir = 'train2014'
        self.data_file = '{}/annotations/instances_{}.json'.format(self.coco_path, self.data_dir)
        self.caption_file = '{}/annotations/captions_{}.json'.format(self.coco_path, self.data_dir)
        # initialize COCO api for image and instance annotations
        self.coco = COCO(self.data_file)
        self.coco_capts = COCO(self.caption_file)
        self.cats = self.coco.loadCats(self.coco.getCatIds())

    def print_progress(self, generated, total):
        print("\r[{0:5.2f}%]".format(float(generated) / float(total) * 100), end="")

    def run(self):
        """
        Return batches of images from the MSCOCO dataset and their respective captions if "captions" is True
        :param num_batches[Int]: number of batches to return
        :param imgs_per_batch[Int]: number of images to per batch return
        :param captions[Bool]: Whether or not to return captions with images
        :param mode[str]: "train" or "val" for training or validation set
        :return: a list of images, optionally a list of captions
        """
        img_ids = []
        all_ids = self.coco.getImgIds()
        for cat in self.cats:
            cat_imgs = self.coco.getImgIds(catIds=cat["id"])
            cat_name = [cat["name"]] * 200
            # print(list(zip(cat_imgs[:20], cat_name)))
            img_ids.extend(zip(cat_imgs[:200], cat_name))

        self.corpus_log_probability = 1.0
        self.accuracy = []
        self.lengths = []

        ### Uncomment for qualitative analysis
        # for cat_id in self.cat_ids:
        #     cat = self.coco.loadCats(ids=cat_id)[0]
        #     cat_imgs = self.coco.getImgIds(catIds=cat_id)
        #     img_batches = np.zeros((self.batch_size, 2048), dtype=np.float32)
        #     cap_batches = []
        #     img_count = 0
        #     for img_id in cat_imgs:
        img_count = 0
        img_batches = np.zeros((self.batch_size, 2048), dtype=np.float32)
        cap_batches = []
        cat_names = []
        for img_id, cat_name in tqdm(img_ids):
            img = self.coco.loadImgs(img_id)[0]
            with open('{}/images/{}/{}'.format(self.coco_path, self.feat_dir, img['file_name']), "rb") as f:
                img = pickle.load(f)

            img_batches[img_count] = img

            img_captions = []
            ann_id = self.coco_capts.getAnnIds(imgIds=img_id)
            anns = self.coco_capts.loadAnns(ann_id)
            for a in anns:
                img_captions.append(self.get_useable_captions(a['caption']))
            cap_batches.append(img_captions)
            cat_names.append(cat_name)

            img_count += 1

            if img_count >= self.batch_size:
                predictions, probabilities = self.run_batch((img_batches, cap_batches))
                print(np.expand_dims(predictions, axis=1).shape)
                self.update_count(cat_names, cap_batches, predicted=False)
                self.update_count(cat_names, np.expand_dims(predictions, axis=1), predicted=True)

                candidates = np.zeros((Agent.D + 1, Agent.batch_size, 2048))
                candidates[0, :] = img_batches
                distractor_ids = np.random.choice(all_ids, Agent.D * Agent.batch_size, replace=False)
                distractor_imgs = self.coco.loadImgs(distractor_ids)
                for d in range(Agent.D):
                    for bs in range(Agent.batch_size):
                        img = distractor_imgs[d * Agent.D + bs]
                        with open('{}/images/{}/{}'.format(self.coco_path, self.feat_dir, img['file_name']), "rb") as f:
                            feats = pickle.load(f)
                            candidates[d + 1, bs] = feats

                self._calculate_omission_score(candidates, predictions)
                self._calculate_bleu_score(predictions, cap_batches)
                self._update_corpus_probability(probabilities, cap_batches)

                img_count = 0
                cap_batches = []
                cat_names = []

    def _calculate_bleu_score(self, pred_ids, true_caps_ids):
        pred_toks = []
        true_caps_toks = []
        for i in range(Agent.batch_size):
            true_caps_toks.append([])
            pt = self.V.ids_to_tokens(pred_ids[i], filter_eos=True)
            pred_toks.append(pt)
            # print(pt)
            self.lengths.append(len(pt.split()))
            for tc in true_caps_ids[i]:
                true_caps_toks[i].append(self.V.ids_to_tokens(tc, filter_eos=True))

        # print("-----")
        # print(pred_toks[0])
        # print(true_caps_toks[0][0])
        # can only do this for sentence level bleu scores (change if we want to report corpus level)
        for ngram, scores in self.bleu_scores.items():
            scores.append(BLEU.get_score(pred_toks, true_caps_toks, N=ngram))
    
    def _calculate_omission_score(self, candidates, messages):
        
        messages = messages.astype(np.int32)
        
        orig_messages = np.zeros((Agent.batch_size, Agent.L, Agent.K))
        
        for i in range(Agent.batch_size):
            orig_messages[i, np.arange(self.L), messages[i]] = 1
            
        # calculate the prob of the target image given the original caption
        fd = {}
        target_indices = np.zeros(self.batch_size, dtype=np.int32)
        self.is_.fill_feed_dict(fd, orig_messages, candidates, target_indices)
        probs, pred = Agent.sess.run([self.is_.prob_dist, self.is_.prediction], feed_dict=fd)
        orig_probs = probs[np.arange(self.batch_size),target_indices]
        orig_probs = orig_probs.reshape((Agent.batch_size, 1))
        accuracy = (Agent.batch_size - np.count_nonzero(pred)) / Agent.batch_size
        self.accuracy.append(accuracy)
        
        mod_probs = []
        for l in range(self.L):
            eos = np.zeros((Agent.batch_size, 1, Agent.K))
            eos[:,:,self.V.eos_id] = 1
            mod_messages = np.concatenate([np.delete(orig_messages, l, axis=1), eos], axis=1)
            
            mod_fd = {}
            target_indices = np.zeros(self.batch_size, dtype=int)
            self.is_.fill_feed_dict(mod_fd, mod_messages, candidates, target_indices)
            curr_mod_probs = Agent.sess.run(is_.prob_dist, feed_dict=mod_fd)[np.arange(self.batch_size),target_indices]
            mod_probs.append(curr_mod_probs)
            
        mod_probs = np.stack(mod_probs, axis=1)
        word_omission_score = orig_probs - mod_probs
        sent_omission_score = np.max(word_omission_score, axis=1)
        self.omission_scores.append(np.mean(sent_omission_score))

    def _update_corpus_probability(self, probabilities, captions):
        for bs in range(Agent.batch_size):
            caption_probabilities = []
            for c in range(len(captions[bs])):
                p = 0.0
                for l in range(Agent.L):
                    p += np.log2(max(probabilities[bs][l][captions[bs][c][l]], 1e-20))
                    if captions[bs][c][l] == self.V.eos_id:
                        break
                caption_probabilities.append(p)
            self.corpus_log_probability += np.mean(caption_probabilities)

    def _perplexity(self, corpus_size):
        return np.power(2.0, -(self.corpus_log_probability / corpus_size))

    def _KL_divergence(self, caption_word_counts, generated_word_counts):
        P = []
        Q = []
        for word, count in caption_word_counts.items():
            P.append(float(generated_word_counts.get(word, 0)))
            Q.append(float(count))

        P = np.array(P) / np.sum(P)
        Q = np.array(Q) / np.sum(Q)
        sum = 0.0
        for i in range(len(P)):
            if P[i] == 0:
                continue
            sum += P[i] * np.log(P[i] / Q[i])
        return sum

    def get_KL_divergence(self):
        all = self._KL_divergence(self.caption_word_counter["all"], self.generated_word_counter["all"])
        per_category = []
        for cat in self.cats:
            kl = self._KL_divergence(self.caption_word_counter[cat["name"]], self.generated_word_counter[cat["name"]])
            per_category.append(kl)
        return all, np.mean(per_category)

    def beam_search(self, images, captions, beam_width=3):
        # BEAM SEARCH

        # Get starting probabilities
        candidate_seq = np.full((Agent.batch_size, 1), Agent.V.sos_id, dtype=np.int32)
        fd = {}
        self.ic.fill_feed_dict(fd, images, candidate_seq, None, seq_len=1)
        prediction, probabilities = Agent.sess.run(self.ops, feed_dict=fd)
        # Create best base sequences
        top_indices = np.argpartition(probabilities, -beam_width)[:, :, -beam_width:]
        tiled_sos_input = np.tile(np.expand_dims(candidate_seq, 0), (beam_width, 1, 1))
        candidate_seq = np.concatenate([tiled_sos_input, np.transpose(top_indices, (2, 0, 1))], axis=2)
        # Set candidate scores
        candidate_scores = np.zeros((beam_width, self.batch_size))
        for bw in range(beam_width):
            candidate_scores[bw] = probabilities[range(self.batch_size), 0, top_indices[:, 0, bw]]

        # For each timestep
        for t in range(2, Agent.L + 1):

            new_candidates = np.zeros((beam_width ** 2, Agent.batch_size, t + 1))
            new_candidate_scores = np.zeros((beam_width ** 2, Agent.batch_size))
            ps = np.zeros((beam_width ** 2, Agent.batch_size, Agent.L, Agent.K), dtype=np.float64)
            nc = 0
            for cs, css in zip(candidate_seq, candidate_scores):
                fd = {}
                self.ic.fill_feed_dict(fd, images, cs, None, seq_len=t)
                prediction, probabilities = Agent.sess.run(self.ops, feed_dict=fd)
                top_indices = np.argpartition(probabilities[:, -1, None, :], -beam_width)[:, :, -beam_width:]

                tiled_seq = np.tile(np.expand_dims(cs, 0), (beam_width, 1, 1))
                candidate_seq = np.concatenate([tiled_seq, np.transpose(top_indices, (2, 0, 1))], axis=2)
                for bw in range(beam_width):
                    ps[nc, :, :t, :] = probabilities
                    new_candidates[nc] = candidate_seq[bw]
                    new_candidate_scores[nc] = css * probabilities[range(self.batch_size), -1, top_indices[:, -1, bw]]
                    nc += 1

            if t >= Agent.L:
                break

            top_indices = np.argpartition(new_candidate_scores, -beam_width, axis=0)[-beam_width:]
            candidate_seq = new_candidates[top_indices, range(self.batch_size)]
            candidate_scores = new_candidate_scores[top_indices, range(self.batch_size)]

        top_index = np.argmax(candidate_scores, axis=0)
        prediction = candidate_seq[top_index, range(self.batch_size), 1:]
        probabilities = ps[top_index, range(self.batch_size)]
        # probabilities = candidate_scores[top_index, range(self.batch_size)]
        return prediction, probabilities
    
    def run_batch(self, inputs, mode="train", beam_search=False, greedy=False):
        """
        Run the Image captioning and return prediction and probability of the sequence
        """
        images, captions = inputs
        images = np.squeeze(images)

        prediction, probabilities = self.beam_search(images, captions, beam_width=3)

        return prediction, probabilities

    def get_useable_captions(self, caption):
        """
        returns captions that can be fed to an image captioner (sequences of ids)
        in_captions are prepended by a sos token, while out_captions are the full sequence.
        Both are capped by the maximum length of a caption
        :param captions:
        :return:
        """
        tokens = caption.translate(str.maketrans('', '', string.punctuation))
        tokens = tokens.lower().split()
        return self.V.tokens_to_ids(self.L, tokens)

    def update_count(self, categories, caption_batches, predicted=True):
        word_counter = self.generated_word_counter if predicted else self.caption_word_counter
        # Over batches
        for i, captions in enumerate(caption_batches):
            category = categories[i]
            # Over possible captions
            for caption in captions:
                # Tokens in captions
                for tok_id in caption:
                    tok = self.V.get_token(tok_id)
                    word_counter[category] = word_counter.get(category, {})
                    word_counter[category][tok] = word_counter[category].get(tok, 0) + 1
                    word_counter["all"] = word_counter.get("all", {})
                    word_counter["all"][tok] = word_counter["all"].get(tok, 0) + 1
                    if tok_id == self.V.eos_id:
                        break

    def get_statistics(self, qualitative=False):
        stats = {}
        stats["corpus_size"] = np.sum(list(self.caption_word_counter["all"].values()))
        stats["vocab_size"] =  len(self.generated_word_counter)
        stats["perplexity"] = self._perplexity(stats["corpus_size"])
        stats["BLEU"] = {
            1: np.mean(self.bleu_scores[1]),
            2: np.mean(self.bleu_scores[2]),
            3: np.mean(self.bleu_scores[3]),
            4: np.mean(self.bleu_scores[4])
        }
        stats["BLEU"]["all"] = np.mean(list(stats["BLEU"].values()))
        stats["omission"] = np.mean(self.omission_scores)
        stats["KL_Divergence"] = self.get_KL_divergence()
        stats["accuracy"] = np.mean(self.accuracy)
        stats["average len"] = np.mean(self.lengths)

        print(stats)

        # for cat in self.generated_word_counter:
        #     if qualitative:
        #         stats[cat] = {"generated": {}, "annotations": {}}
        #         gen_c = Counter(self.generated_word_counter[cat])
        #         ann_c = Counter(self.annotation_word_counter[cat])
        #
        #         for key, counter in zip(["generated", "annotations"], [gen_c, ann_c]):
        #             stats[cat][key]["top10"] = counter.most_common(10)
        #             stats[cat][key]["tokens_used"] = len(counter)
        #
        #         print("========== {} ==========".format(cat))
        #         print("The annotations contain {} unique words, the agent generated {} unique words"
        #               .format(stats[cat]["annotations"]["tokens_used"], stats[cat]["generated"]["tokens_used"]))
        #         print("The top 10 most frequently used works for images containing {} are:".format(cat))
        #         for i in range(10):
        #             g, gc = stats[cat]["generated"]["top10"][i]
        #             a, ac = stats[cat]["annotations"]["top10"][i]
        #             print("Annotations: {} used {} times, Generated: {} used {} times".format(a, ac, g, gc))
        return stats

if __name__ == "__main__":




    load_key = "7e989143b2d94b2ba262496c203f7836" #""9dfc5e42bae84c7689708e3631b3c630"

    Agent.set_params(K=10000, D=1, L=15, batch_size=128, train=False, loss_type='pairwise')

    with tf.variable_scope("all", reuse=tf.AUTO_REUSE):
        ic = ImageCaptioner(load_key=load_key)
        is_ = ImageSelector(load_key=load_key)
        
        # Initialize TF
        variables_to_initialize = tf.global_variables()
        dont_initialize = ImageCaptioner.get_all_weights() + ImageSelector.get_all_weights()
        variables_to_initialize = [v for v in tf.global_variables() if v not in dont_initialize]
        Agent.sess.run(tf.variables_initializer(variables_to_initialize))
        e = Evaluator(ic, is_)

        e.run()
        e.get_statistics()



