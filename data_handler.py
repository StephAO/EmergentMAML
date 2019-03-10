import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from skimage import io, transform

# TODO: consider moving this to a better spot
img_h = 96
img_w = 96

class Data_Handler:

    def __init__(self, dataset=None):
        self.data_dir = '/home/stephane/cocoapi'
        self.dataType = 'train2014'
        self.data_file = '{}/annotations/instances_{}.json'.format(self.data_dir, self.dataType)
        self.caption_file = '{}/annotations/captions_{}.json'.format(self.data_dir, self.dataType)
        # initialize COCO api for image and instance annotations
        self.coco = COCO(self.data_file)
        # Uncomment to enable captions
        # self.coco_capts = COCO(self.caption_file)
        # COCO categories
        # self.cats = self.coco.loadCats(self.coco.getCatIds())
        self.catIds = self.coco.getCatIds()

        self.train_split = 0.8

    def split_train_val(self):
        self.val = {}
        self.train = {}
        for cat_id in self.catIds:
            imgIds = self.coco.getImgIds(catIds=cat_id)
            split_idx = int(len(imgIds * self.train_split))
            self.train[cat_id] = imgIds[:split_idx]
            self.val[cat_id] = imgIds[split_idx:]

    def get_images(self, imgs_per_batch=1, num_batches=1, captions=False, data_type="train"):
        """
        Return batches of images from the MSCOCO dataset and their respective captions if "captions" is True
        :param num_batches[Int]: number of batches to return
        :param imgs_per_batch[Int]: number of images to per batch return
        :param captions[Bool]: Whether or not to return captions with images
        :param data_type[str]: "train" or "val" for training or validation set
        :return: a list of images, optionally a list of captions
        """
        data = self.train if data_type == "train" else self.val
        data_idx = {}
        for cat in self.catIds:
            np.random.shuffle(data[cat])
            data_idx[cat] = len(data[cat] - 1)

        while True:
            img_batches = np.zeros((imgs_per_batch, num_batches, img_h, img_w, 3), dtype=np.float32)
            cap_batches = []
            for b in range(num_batches):
                if img_batches > len(data_idx):
                    break
                cat_idxs = np.random.randint(len(data_idx), size=imgs_per_batch)
                cat_ids = [data_idx.keys()[i] for i in cat_idxs]

                annotations = []
                for i, cat in enumerate(cat_ids):
                    img_id = data[cat][data_idx[cat]]
                    img = self.coco.loadImgs(img_id)[0]
                    data_idx[cat] -= 1
                    if data_idx[cat] <= 0:
                        data_idx.pop(cat)
                    img = io.imread('{}/images/{}/{}'.format(self.data_dir, self.dataType, img['file_name']))

                    # Increase channels (by copying) of bw images that don't have 3 channels
                    while img.shape[-1] != 3:
                        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

                    img = transform.resize(img, (img_h, img_w), anti_aliasing=True, mode='reflect')

                    # plt.axis('off')
                    # plt.imshow(img)
                    # plt.show()

                    img_batches[i, b] = img
                    if captions:
                        ann_id = self.coco_capts.getAnnIds(imgIds=img_id)
                        anns = self.coco_capts.loadAnns(ann_id)
                        annotations.append(anns)

                if captions:
                    cap_batches.append(annotations)

            ret = (img_batches, cap_batches) if captions else img_batches
            yield ret
