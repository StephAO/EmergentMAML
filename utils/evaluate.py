import numpy as np
import matplotlib.pyplot as plt
import os
from pycocotools.coco import COCO
import pickle
import string
import tensorflow as tf
from collections import Counter

# TODO: consider moving this to a better spot
from utils.data_handler import coco_path, project_path
from Agents import Agent, ImageCaptioner

class Evaluator:

    def __init__(self, image_captioner):
        self.ic = image_captioner
        self.ops = self.ic.get_output()
        self.batch_size = Agent.batch_size
        self.V = Agent.V
        self.L = Agent.L
        # Word counters
        self.generated_word_counter = {}
        self.annotation_word_counter = {}

        # MSCOCO variables
        self.coco_path = coco_path
        self.feat_dir = 'train_feats'
        self.data_dir = 'train2014'
        self.data_file = '{}/annotations/instances_{}.json'.format(self.coco_path, self.data_dir)
        self.caption_file = '{}/annotations/captions_{}.json'.format(self.coco_path, self.data_dir)
        # initialize COCO api for image and instance annotations
        self.coco = COCO(self.data_file)
        self.coco_capts = COCO(self.caption_file)
        self.cat_ids = self.coco.getCatIds()

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
        generated = 0
        total = 0
        for cat_id in self.cat_ids:
            cat_imgs = self.coco.getImgIds(catIds=cat_id)
            total += len(cat_imgs)

        for cat_id in self.cat_ids:
            cat = self.coco.loadCats(ids=cat_id)[0]
            cat_imgs = self.coco.getImgIds(catIds=cat_id)
            img_batches = np.zeros((self.batch_size, 2048), dtype=np.float32)
            cap_batches = []
            img_count = 0
            for img_id in cat_imgs:
                img = self.coco.loadImgs(img_id)[0]
                with open('{}/images/{}/{}'.format(self.coco_path, self.feat_dir, img['file_name']), "rb") as f:
                    img = pickle.load(f)

                img_batches[img_count] = img

                img_captions = []
                ann_id = self.coco_capts.getAnnIds(imgIds=img_id)
                anns = self.coco_capts.loadAnns(ann_id)
                for a in anns:
                    img_captions.append(a['caption'])
                cap_batches.append(img_captions)

                img_count += 1

                if img_count >= self.batch_size:
                    captions, predictions = self.run_batch((img_batches, cap_batches))
                    self.update_count(cat["name"], captions, predicted=False)
                    self.update_count(cat["name"], predictions, predicted=True)
                    img_count = 0
                    cap_batches = []

                    generated += self.batch_size
                    self.print_progress(generated, total)


    def run_batch(self, inputs, mode="train"):
        """
        Run the Image captioning to learn parameters
        """
        images, captions = inputs
        images = np.squeeze(images)

        in_captions, out_captions = self.get_useable_captions(captions)

        fd = {}
        self.ic.fill_feed_dict(fd, images, in_captions, out_captions)
        accuracy, loss, prediction = Agent.sess.run(self.ops, feed_dict=fd)

        return out_captions, prediction

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
            chosen_caption = caption[np.random.randint(5)]
            tokens = chosen_caption.translate(str.maketrans('', '', string.punctuation))
            tokens = tokens.lower().split()
            out_captions[i] = self.V.tokens_to_ids(self.L, tokens)
            in_captions[i] = np.roll(out_captions[i], 1)
            in_captions[i][0] = self.V.sos_id

        return in_captions, out_captions

    def update_count(self, category, sentences, predicted=True):
        word_counter = self.generated_word_counter if predicted else self.annotation_word_counter
        for s in sentences:
            for id in s:
                tok = self.V.get_token(id)
                if tok in ["</s>","a","the","on","of","in","with","and","is","are","to","an","at","it"]:
                    continue

                word_counter[category] = word_counter.get(category, {})
                word_counter[category][tok] = word_counter[category].get(tok, 0) + 1

    def get_statistics(self):
        stats = {}
        for cat in self.generated_word_counter:
            stats[cat] = {"generated": {}, "annotations": {}}
            gen_c = Counter(self.generated_word_counter[cat])
            ann_c = Counter(self.annotation_word_counter[cat])

            for key, counter in zip(["generated", "annotations"], [gen_c, ann_c]):
                stats[cat][key]["top10"] = counter.most_common(10)
                stats[cat][key]["tokens_used"] = len(counter)

            print("========== {} ==========".format(cat))
            print("The annotations contain {} unique words, the agent generated {} unique words"
                  .format(stats[cat]["annotations"]["tokens_used"], stats[cat]["generated"]["tokens_used"]))
            print("The top 10 most frequently used works for images containing {} are:".format(cat))
            for i in range(10):
                g, gc = stats[cat]["generated"]["top10"][i]
                a, ac = stats[cat]["annotations"]["top10"][i]
                print("Annotations: {} used {} times, Generated: {} used {} times".format(a, ac, g, gc))


if __name__ == "__main__":
    load_key = "3bf9625ce5ee4bdda514aa4b40ae72db"

    Agent.set_params(K=10000, D=1, L=15, batch_size=128, train=False)

    with tf.variable_scope("all", reuse=tf.AUTO_REUSE):
        ic = ImageCaptioner(load_key=load_key)
        # Initialize TF
        variables_to_initialize = tf.global_variables()
        dont_initialize = ImageCaptioner.get_all_weights()
        variables_to_initialize = [v for v in tf.global_variables() if v not in dont_initialize]
        Agent.sess.run(tf.variables_initializer(variables_to_initialize))
        e = Evaluator(ic)
        e.run()
        e.get_statistics()



