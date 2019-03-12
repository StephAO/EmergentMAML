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
        self.coco_capts = COCO(self.caption_file)
        # COCO categories
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self.catIds = self.coco.getCatIds() #catNms=["elephant", "airplane"])

    def get_images(self, imgs_per_batch=1, num_batches=1, return_captions=False):
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
            cat_idxs = np.random.randint(len(self.catIds), size=imgs_per_batch)
            catIds = [self.catIds[i] for i in cat_idxs]

            captions = []
            for i, cat in enumerate(catIds):
                # catId = self.coco.getCatIds(catNms=[cat["name"]])
                imgIds = self.coco.getImgIds(catIds=cat)#["id"])
                # Testing using same two images. Commented code is used for random image (harder but that's the goal)
                
                # TODO - check that this correction is correct
                imgId = imgIds[np.random.randint(0, len(imgIds))]
                
                img = self.coco.loadImgs(imgId)[0]
                img = io.imread('{}/images/{}/{}'.format(self.data_dir, self.dataType, img['file_name']))


                # Ignore images that don't have 3 channels
                # TODO: consider making bw images 3 channels to make them useable
                while img.shape[-1] != 3:
                    imgId = imgIds[np.random.randint(0, len(imgIds))]
                    img = self.coco.loadImgs(imgId)[0]
                    img = io.imread('{}/images/{}/{}'.format(self.data_dir, self.dataType, img['file_name']))

                img = transform.resize(img, (img_h, img_w), anti_aliasing=True, mode='reflect')

                # plt.axis('off')
                # plt.imshow(img)
                # plt.show()

                img_batches[i, b] = img
                if return_captions:
                    annIds = self.coco_capts.getAnnIds(imgIds=[imgId])
                    anns = self.coco_capts.loadAnns(annIds)
                    for a in anns:
                        captions.append(a['caption'])

            if return_captions:
                cap_batches.append(captions)

        ret = (img_batches, cap_batches) if return_captions else img_batches
        return ret