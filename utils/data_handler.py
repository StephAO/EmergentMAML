import numpy as np
import matplotlib.pyplot as plt
import os
from pycocotools.coco import COCO
import pickle
import tensorflow as tf

# TODO: consider moving this to a better spot
coco_path =  '/h/stephaneao/cocoapi'
project_path = '/h/stephaneao/EmergentMAML'

class Data_Handler:

    def __init__(self, num_distractors=0, batch_size=1, same_category=True):
        self.num_distractors = num_distractors
        self.batch_size = batch_size
        self.coco_path = coco_path
        self.feat_dir = 'new_train_feats'#new_train_feats
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
        self.same_category = same_category
        self.split_train_val()

    def get_params(self):
        """
        Returns a dictionary of parameters to track.
        """
        params = {
            "train_split": self.train_split,
            "grouping" : self.same_category,
        }

        return params

    def split_train_val(self):
        all_img_ids = self.coco.getImgIds()
        np.random.seed(12345)
        np.random.shuffle(all_img_ids)
        self.all_imgs = {"train": [], "val": []}
        self.all_imgs_ids = {"train": [], "val": []}

        # All images
        # Split images into train/val
        split_idx = int(len(all_img_ids) * self.train_split)

        # Each set of images is grouped in a way (usually where each has a different category)

        self.distractors_cat = {"train": {}, "val": {}}

        for i, img_id in enumerate(all_img_ids):
            mode = "train" if i < split_idx else "val"
            ann_id = self.coco.getAnnIds(img_id)
            anns = self.coco.loadAnns(ann_id)

            categories = []
            for a in anns:
                cat_id = a['category_id']
                categories.append(cat_id)
                if cat_id not in self.distractors_cat[mode]:
                    self.distractors_cat[mode][cat_id] = []
                self.distractors_cat[mode][cat_id].append(img_id)
            if len(categories) == 0:
                if -1 not in self.distractors_cat[mode]:
                    self.distractors_cat[mode][-1] = []
                self.distractors_cat[mode][-1].append(img_id)

            self.all_imgs[mode].append((img_id, tuple(categories)))
            self.all_imgs_ids[mode].append(img_id)

    def set_params(self, distractors=None, batch_size=None):
        self.num_distractors = self.num_distractors if distractors is None else distractors
        self.batch_size = batch_size or self.batch_size

    def get_images(self, return_captions=False, mode="train"):
        """
        Return batches of images from the MSCOCO dataset and their respective captions if "captions" is True
        :param num_batches[Int]: number of batches to return
        :param imgs_per_batch[Int]: number of images to per batch return
        :param captions[Bool]: Whether or not to return captions with images
        :param mode[str]: "train" or "val" for training or validation set
        :return: a list of images, optionally a list of captions
        """
        mode = "train" if mode[-5:] == "train" else "val"

        data_idx = 0
        np.random.shuffle(self.all_imgs[mode])

        while data_idx + self.batch_size < len(self.all_imgs[mode]):
            img_batches = np.zeros((1 + self.num_distractors, self.batch_size, 2048), dtype=np.float32)
            cap_batches = []
            for b in range(self.batch_size):
                img_id, categories = self.all_imgs[mode][data_idx]
                data_idx += 1

                if return_captions:
                    img_captions = []
                    ann_id = self.coco_capts.getAnnIds(imgIds=img_id)
                    anns = self.coco_capts.loadAnns(ann_id)
                    for a in anns:
                        img_captions.append(a['caption'])
                    cap_batches.append(img_captions)

                img = self.coco.loadImgs(img_id)[0]
                with open('{}/images/{}/{}'.format(self.coco_path, self.feat_dir, img['file_name']), "rb") as f:
                    img = pickle.load(f)

                img_batches[0, b] = img

                # Get distractors
                if self.same_category:
                    distractor_options = []
                    if len(categories) == 0:
                        distractor_options += self.distractors_cat[mode][-1]
                    else:
                        for category in categories:
                            distractor_options += self.distractors_cat[mode][category]
                else:
                    distractor_options = self.all_imgs_ids[mode]

                distractor_ids = np.random.choice(distractor_options, self.num_distractors, replace=False)
                while img_id in distractor_ids:
                    idx = np.where(distractor_ids == img_id)
                    distractor_ids[idx] = np.random.choice(distractor_options, 1, replace=False)
                distractor_imgs = self.coco.loadImgs(distractor_ids)
                for i, d in enumerate(distractor_imgs):
                    with open('{}/images/{}/{}'.format(self.coco_path, self.feat_dir, d['file_name']), "rb") as f:
                        feats = pickle.load(f)
                    img_batches[i + 1, b] = feats

            ret = (img_batches, cap_batches) if return_captions else img_batches
            yield ret

    def generate_all_encodings(self):

        #  Shared CNN pre-trained on imagenet, see https://github.com/keras-team/keras-applications for other options

        pre_trained = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet',
                                                                     pooling='max',
                                                                     input_shape=(299, 299, 3))

        # Create save/load directory
        dir = '{}/images/{}'.format(self.coco_path, self.feat_dir)
        if not os.path.exists(dir):
            os.makedirs(dir)

        img_ph = tf.placeholder(tf.float32)
        img_feats = pre_trained(img_ph)
        sess = tf.keras.backend.get_session()#tf.Session()

        # sess.run(tf.variables_initializer(tf.global_variables()))

        count = 0

        img_list = list(self.coco.getImgIds())

        batch_feat = []
        batch_img = []

        for img_id in img_list:
            img = self.coco.loadImgs(img_id)[0]
            img_fn = img['file_name']
            img_path = '{}/images/{}/{}'.format(self.coco_path, self.data_dir, img_fn)
            img = tf.keras.preprocessing.image.load_img(img_path, color_mode='rgb', target_size=(299, 299))

            img = tf.keras.preprocessing.image.img_to_array(img)

            img = np.expand_dims(img, axis=0)

            # Increase channels (by copying) of bw images that don't have 3 channels
            while img.shape[-1] != 3:
                assert False

            # set image data between -1 and 1
            img /= 255.
            img -= 0.5
            img *= 2.

            img_feat = sess.run([img_feats], feed_dict={img_ph: img})

            batch_feat.append(img_feat)
            batch_img.append(img)
            if len(batch_img) >= 128:
                batch_feat = []
                batch_img = []

            img_feat = np.squeeze(img_feat)


            with open('{}/images/{}/{}'.format(self.coco_path, self.feat_dir, img_fn), "wb") as f:
                pickle.dump(img_feat, f)

            count += 1


        self.print_progress(count, len(img_list))

if __name__ == "__main__":
    dh = Data_Handler()
    dh.generate_all_encodings()

