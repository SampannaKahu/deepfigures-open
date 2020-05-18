import os
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

import tensorflow_datasets as tfds

tfds.load(
    name='coco',
    split=tfds.Split.TRAIN,
    # data_dir='/home/sampanna/deepfigures-results',
    data_dir='/home/sampanna/tmp_coco_dataset',
    batch_size=1,
    shuffle_files=False,
    # download=False,
    download=True,
    as_supervised=False
)
