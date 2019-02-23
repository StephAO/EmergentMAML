import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from skimage import io, transform

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
        self.cats = self.coco.loadCats(self.coco.getCatIds())

    def get_images(self, imgs_per_batch=1, num_batches=1, captions=False):
        """
        Return batches of images from the MSCOCO dataset and their respective captions if "captions" is True
        :param num_batches[Int]: number of batches to return
        :param imgs_per_batch[Int]: number of images to per batch return
        :param captions[Bool]: Whether or not to return captions with images
        :return: a list of images, optionally a list of captions
        """
        img_batches = np.zeros((imgs_per_batch, num_batches, img_h, img_w, 3), dtype=np.float32)
        cap_batches = []
        for b in range(num_batches):
            # categories = np.random.choice(self.cats, n)
            # Testing using the same two categories to make things easier, uncomment to improve this
            catIds = self.coco.getCatIds(catNms=['airplane', 'elephant'])
            annotations = []
            for i, cat in enumerate(catIds):
                # catId = self.coco.getCatIds(catNms=[cat["name"]])
                imgIds = self.coco.getImgIds(catIds=cat)#["id"])
                # Testing using same two images. Commented code is used for random image (harder but that's the goal)
                img = self.coco.loadImgs(imgIds[0])[0]#imgIds[np.random.randint(0, len(imgIds))])[0]
                img = io.imread('{}/images/{}/{}'.format(self.data_dir, self.dataType, img['file_name']))


                # Ignore images that don't have 3 channels
                while img.shape[-1] != 3:
                    print("bw_image", img.shape)

                    img = self.coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
                    img = io.imread('{}/images/{}/{}'.format(self.data_dir, self.dataType, img['file_name']))

                img = transform.resize(img, (img_h, img_w), anti_aliasing=True, mode='reflect')

                # Making the problem easier to debug
                # if i == 0:
                #     img = np.zeros((img_h, img_w, 3), dtype=np.float32)
                # else:
                #     img = np.ones((img_h, img_w, 3), dtype=np.float32)

                # plt.axis('off')
                # plt.imshow(img)
                # plt.show()

                img_batches[i, b] = img
                if captions:
                    annIds = self.coco_capts.getAnnIds(imgIds=imgIds)
                    anns = self.coco_capts.loadAnns(annIds)
                    annotations.append(anns)

            if captions:
                cap_batches.append(annotations)

        ret = (img_batches, cap_batches) if captions else img_batches
        return ret