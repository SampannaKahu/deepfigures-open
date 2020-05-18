import os
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from torchvision.datasets import CocoDetection
from torchvision import transforms
from deepfigures import settings
from torch.utils.data import DataLoader


class TransformedCocoDataset(CocoDetection):
    def __init__(self, root, annFile, iaa_pipeline: iaa.Sequential = None):
        super(TransformedCocoDataset, self).__init__(root, annFile, transform=None,
                                                     target_transform=None, transforms=None)
        self.iaa_pipeline = iaa_pipeline

    def __getitem__(self, i):
        image, annos = super(TransformedCocoDataset, self).__getitem__(i)
        bbs = [ia.BoundingBox(x1=anno['segmentation'][0][0],
                              y1=anno['segmentation'][0][1],
                              x2=anno['segmentation'][0][2],
                              y2=anno['segmentation'][0][7]) for anno in annos]
        _images, _bbs = self.iaa_pipeline(images=[np.array(image)], bounding_boxes=[bbs])
        return transforms.ToTensor()(_images[0]), transforms.ToTensor()(
            np.array([[bb.x1, bb.y1, bb.x2, bb.y2] for bb in _bbs[0]]))


dataset = TransformedCocoDataset(root='/home/sampanna/deepfigures-results/arxiv_coco_dataset/images',
                                 annFile='/home/sampanna/deepfigures-results/arxiv_coco_dataset/annotations.json',
                                 iaa_pipeline=settings.seq)

dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

for images, annos in dataloader:
    print(images, annos)
