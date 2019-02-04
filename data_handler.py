import numpy as np
from pycocotools.coco import COCO



class Data_Handler:

    def __init__(self):
        self.data_dir = '~/cocoapi'
        self.dataType = 'train2014'
        self.data_file = '{}/annotations/instances_{}.json'.format(self.data_dir, self.dataType)
        self.caption_file = '{}/annotations/captions_{}.json'.format(self.data_dir, self.dataType)
        # initialize COCO api for image and instance annotations
        self.coco = COCO(self.data_file)
        self.coco_capts = COCO(self.caption_file)
        # COCO categories
        self.cats = self.coco.loadCats(self.coco.getCatIds())

    def get_images(self, num=1, captions=False):
        """
        Return "num" number of images from the MSCOCO dataset and their respective captions if "captions" is True
        :param num: number of images to return
        :param captions[Bool]: Whether or not to return captions with images
        :return: a list of images, optionally a list of captions
        """
        categories = np.random.choice(self.cats.keys(), num)
        images = []
        annotations = []
        for cat_key in categories:
            catId = self.coco.getCatIds(catNms=[self.cats[cat_key]["name"]])
            imgIds = self.coco.getImgIds(catIds=catId)
            img = self.coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
            images.append(img)

            if captions:
                annIds = self.coco_capts.getAnnIds(imgIds=imgIds)
                anns = self.coco_capts.loadAnns(annIds)
                annotations.append(anns)

        if captions:
            return images, annotations
        else:
            return images