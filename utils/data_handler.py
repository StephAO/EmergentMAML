import numpy as np
import matplotlib.pyplot as plt
import os
from pycocotools.coco import COCO
import pickle
import tensorflow as tf
from tqdm import tqdm

# TODO: consider moving this to a better spot
coco_path = '/home/stephane/cocoapi'
project_path = '/home/stephane/PycharmProjects/EmergentMAML'

class Data_Handler:

    def __init__(self, images_per_instance=1, batch_size=1, group=True):
        self.images_per_instance = images_per_instance
        self.batch_size = batch_size
        self.images_per_batch = self.images_per_instance * self.batch_size
        self.coco_path = coco_path
        self.feat_dir = 'train_feats'
        self.data_dir = 'train2014'
        self.data_file = '{}/annotations/instances_{}.json'.format(self.coco_path, self.data_dir)
        self.caption_file = '{}/annotations/captions_{}.json'.format(self.coco_path, self.data_dir)
        # initialize COCO api for image and instance annotations
        self.coco = COCO(self.data_file)
        # Uncomment to enable captions
        self.coco_capts = COCO(self.caption_file)
        # COCO categories
        # self.cats = self.coco.loadCats(self.coco.getCatIds())
        self.cat_ids = self.coco.getCatIds()

        self.train_split = 0.90
        self.group = group
        self.split_train_val()

    def get_params(self):
        """
        Returns a dictionary of parameters to track.
        """
        params = {
            "train_split": self.train_split,
            "grouping" : self.group,
        }

        return params

    def split_train_val(self):
        all_img_ids = self.coco.getImgIds()
        np.random.shuffle(all_img_ids)

        # No grouping when selecting distractors
        if not self.group:
            # Split images into train/val
            split_idx = int(len(all_img_ids) * self.train_split)
            self.train = all_img_ids[:split_idx]
            self.val = all_img_ids[split_idx:]
            return

        # Each set of images is grouped in a way (usually where each has a different category)
        self.all = {}
        self.train = {}
        self.val = {}

        for img_id in all_img_ids:
            ann_id = self.coco.getAnnIds(img_id)
            anns = self.coco.loadAnns(ann_id)

            # Find category to place img in
            # Currently using category that takes up the most space in the image
            possible_categories = {}
            if len(anns) == 0:  # If image has not category, place in "other" category bin
                possible_categories[-1] = 1.
            for i in range(len(anns)):
                entity_id = anns[i]['category_id']
                possible_categories[entity_id] = possible_categories.get(entity_id, 0.0) + anns[i]['area']
            category = max(possible_categories, key=lambda key: possible_categories[key])
            if not category in self.all:
                self.all[category] = []
            self.all[category].append(img_id)

        # Split images into train/val
        for cat, imgs in self.all.items():
            split_idx = int(len(imgs) * self.train_split)
            self.train[cat] = self.all[cat][:split_idx]
            self.val[cat] = self.all[cat][split_idx:]

    def set_params(self, images_per_instance=None, batch_size=None):
        self.images_per_instance = images_per_instance or self.images_per_instance
        self.batch_size = batch_size or self.batch_size
        self.images_per_batch = self.images_per_instance * self.batch_size

    def print_progress(self, generated, total):
        print("\r[{0:5.2f}%]".format(float(generated) / float(total) * 100), end="")

    def get_images(self, return_captions=False, mode="train"):
        """
        Return batches of images from the MSCOCO dataset and their respective captions if "captions" is True
        :param num_batches[Int]: number of batches to return
        :param imgs_per_batch[Int]: number of images to per batch return
        :param captions[Bool]: Whether or not to return captions with images
        :param mode[str]: "train" or "val" for training or validation set
        :return: a list of images, optionally a list of captions
        """
        data = self.train if mode == "train" else self.val
        generated = 0
        total = 0
        if self.group:
            data_idx = {}
            for cat in self.cat_ids:
                np.random.shuffle(data[cat])
                data_idx[cat] = len(data[cat]) - 1
                total += len(data[cat])
        else:
            data_idx = 0
            np.random.shuffle(data)
            total += len(data)

        while self.group or data_idx + self.images_per_batch < len(data):
            img_batches = np.zeros((self.images_per_instance, self.batch_size, 2048), dtype=np.float32)
            cap_batches = []
            for b in range(self.batch_size):
                if self.group and self.images_per_instance > len(data_idx):
                    self.print_progress(generated, total)
                    return
                elif self.group:
                    cat_ids = np.random.choice(list(data_idx.keys()), replace=False, size=self.images_per_instance)

                gen = enumerate(cat_ids) if self.group else range(self.images_per_instance)

                captions = []
                for i in gen:
                    if self.group:
                        i, cat = i
                        img_id = data[cat][data_idx[cat]]
                        data_idx[cat] -= 1
                        if data_idx[cat] < 0:
                            data_idx.pop(cat)
                    else:
                        img_id = data[data_idx]
                        data_idx += 1

                    if return_captions:
                        img_captions = []
                        ann_id = self.coco_capts.getAnnIds(imgIds=img_id)
                        anns = self.coco_capts.loadAnns(ann_id)
                        for a in anns:
                            img_captions.append(a['caption'])
                        captions.append(img_captions)

                    img = self.coco.loadImgs(img_id)[0]
                    with open('{}/images/{}/{}'.format(self.coco_path, self.feat_dir, img['file_name']), "rb") as f:
                        img = pickle.load(f)

                    img_batches[i, b] = img

                if return_captions:
                    cap_batches.append(captions)

            ret = (img_batches, cap_batches) if return_captions else img_batches
            generated += self.images_per_batch
            self.print_progress(generated, total)
            yield ret

        self.print_progress(generated, total)

    def generate_all_encodings(self):
        #  Shared CNN pre-trained on imagenet, see https://github.com/keras-team/keras-applications for other options
        pre_trained = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet')

        # Create save/load directory
        dir = '{}/images/'.format(self.coco_path)
        if not os.path.exists(dir):
            os.makedirs(dir)

        image_features = {}

        img_ph = tf.placeholder(tf.float32)
        pre_img = tf.keras.applications.inception_v3.preprocess_input(img_ph)
        img_feats = pre_trained(pre_img)

        sess = tf.Session()
        batch_size = 64

        sess.run(tf.variables_initializer(tf.global_variables()))

        img_list = list(self.coco.getImgIds())

        images = np.zeros((batch_size, 299, 299, 3))
        image_names = []
        total = len(img_list)

        b = 0

        for i, img_id in tqdm(enumerate(img_list)):

            img = self.coco.loadImgs(img_id)[0]
            img_fn = img['file_name']
            img_path = '{}/images/{}/{}'.format(self.coco_path, self.data_dir, img_fn)

            img = tf.keras.preprocessing.image.load_img(img_path, color_mode='rgb', target_size=(299, 299))
            img = tf.keras.preprocessing.image.img_to_array(img)

            images[b] = img
            image_names.append(img_fn)
            b += 1

            if b == batch_size:

                img_feat = sess.run([img_feats], feed_dict={img_ph: images})

                for i, img in enumerate(img_feat):
                    image_features[image_names[i]] = img

                b = 0
                image_names = []

                if i != 0 and i % (int(total / 10) + 1) == 0:
                    pb = int(i / (int(total / 10)) + 1)
                    print(len(image_features))
                    print(image_features.values()[0].shape)
                    with open('{}/images/{}{}{}'.format(self.coco_path, 'image_features_', pb, '.p'), "wb") as f:
                        pickle.dump(image_features, f)
                        image_features = {}

        pb = int(total / i) + 1
        print(len(image_features))
        print(image_features.values()[0].shape)
        with open('{}/images/{}{}{}'.format(self.coco_path, 'image_features_', pb, '.p'), "wb") as f:
            pickle.dump(image_features, f)






if __name__ == "__main__":
    dh = Data_Handler()
    dh.generate_all_encodings()

