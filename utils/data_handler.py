import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from skimage import io, transform
import time

# TODO: consider moving this to a better spot
img_h = 96
img_w = 96

class Data_Handler:

    def __init__(self, dataset=None):
        self.data_dir = '/home/marina/coco'
        self.dataType = 'train2014'
        self.data_file = '{}/annotations/instances_{}.json'.format(self.data_dir, self.dataType)
        self.caption_file = '{}/annotations/captions_{}.json'.format(self.data_dir, self.dataType)
        # initialize COCO api for image and instance annotations
        self.coco = COCO(self.data_file)
        # Uncomment to enable captions
        self.coco_capts = COCO(self.caption_file)
        # COCO categories
        # self.cats = self.coco.loadCats(self.coco.getCatIds())
        self.cat_ids = self.coco.getCatIds()

        self.train_split = 0.95
        self.group = True
        self.split_train_val()

    def split_train_val(self):
        st = time.time()
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

    def print_progress(self, generated, total):
        print("\r[{0:5.2f}%]".format(float(generated) / float(total) * 100), end="")

    def get_images(self, images_per_instance=1, batch_size=1, return_captions=False, data_type="train"):
        """
        Return batches of images from the MSCOCO dataset and their respective captions if "captions" is True
        :param num_batches[Int]: number of batches to return
        :param imgs_per_batch[Int]: number of images to per batch return
        :param captions[Bool]: Whether or not to return captions with images
        :param data_type[str]: "train" or "val" for training or validation set
        :return: a list of images, optionally a list of captions
        """
        data = self.train if data_type == "train" else self.val
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

        images_per_batch = images_per_instance * batch_size
        while self.group or data_idx + images_per_batch < len(data):
            img_batches = np.zeros((images_per_instance, batch_size, img_h, img_w, 3), dtype=np.float32)
            cap_batches = []
            for b in range(batch_size):
                if self.group and images_per_instance > len(data_idx):
                    self.print_progress(generated, total)
                    return
                elif self.group:
                    cat_ids = np.random.choice(list(data_idx.keys()), replace=False, size=images_per_instance)

                gen = enumerate(cat_ids) if self.group else range(images_per_instance)

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
                    img = self.coco.loadImgs(img_id)[0]
                    img = io.imread('{}/images/{}/{}'.format(self.data_dir, self.dataType, img['file_name']))

                    # Increase channels (by copying) of bw images that don't have 3 channels
                    while img.shape[-1] != 3:
                        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

                    img = transform.resize(img, (img_h, img_w), anti_aliasing=True, mode='reflect')

                    # plt.axis('off')
                    # plt.imshow(img)
                    # plt.show()
                    
                    img_batches[i, b] = img
                    if return_captions:
                        ann_id = self.coco_capts.getAnnIds(imgIds=img_id)
                        anns = self.coco_capts.loadAnns(ann_id)
                        img_captions = []
                        for a in anns:
                            img_captions.append(a['caption'])
                        captions.append(img_captions)

                if return_captions:
                    cap_batches.append(captions)

            ret = (img_batches, cap_batches) if return_captions else img_batches
            generated += images_per_batch
            self.print_progress(generated, total)
            yield ret

        self.print_progress(generated, total)
