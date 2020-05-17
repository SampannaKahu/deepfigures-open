import os
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

import tensorflow_datasets as tfds

tfds.load(
    name='coco',
    split=tfds.Split.TRAIN,
    data_dir='/home/sampanna/deepfigures-results',
    batch_size=1,
    shuffle_files=False,
    download=False,
    as_supervised=False
)
