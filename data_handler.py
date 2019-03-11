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
        self.cat_ids = self.coco.getCatIds()

        self.train_split = 0.95
        self.split_train_val()

    def split_train_val(self):
        all_img_ids = self.coco.getImgIds()
        np.random.shuffle(all_img_ids)

        # TODO consider grouping by categories
        split_idx = int(len(all_img_ids) * self.train_split)
        self.train = all_img_ids[:split_idx]
        self.val = all_img_ids[split_idx:]

    def get_images(self, images_per_instance=1, batch_size=1, captions=False, data_type="train"):
        """
        Return batches of images from the MSCOCO dataset and their respective captions if "captions" is True
        :param num_batches[Int]: number of batches to return
        :param imgs_per_batch[Int]: number of images to per batch return
        :param captions[Bool]: Whether or not to return captions with images
        :param data_type[str]: "train" or "val" for training or validation set
        :return: a list of images, optionally a list of captions
        """
        data = self.train if data_type == "train" else self.val
        data_idx = 0
        np.random.shuffle(data)

        images_per_batch = images_per_instance * batch_size
        while data_idx + images_per_batch < len(data):
            img_batches = np.zeros((images_per_instance, batch_size, img_h, img_w, 3), dtype=np.float32)
            cap_batches = []
            for b in range(batch_size):
                annotations = []
                for i in range(images_per_instance):
                    img_id = data[data_idx]
                    img = self.coco.loadImgs(img_id)[0]
                    data_idx += 1
                    img = io.imread('{}/images/{}/{}'.format(self.data_dir, self.dataType, img['file_name']))

                    # Increase channels (by copying) of bw images that don't have 3 channels
                    while img.shape[-1] != 3:
                        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
                        # plt.axis('off')
                        # plt.imshow(img)
                        # plt.show()

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
            print("\r[{0:5.2f}%]".format(float(data_idx) / float(len(data)) * 100), end="")
            yield ret

        print("\r[{0:5.2f}%]".format(float(len(data)) / float(len(data)) * 100), end="")
