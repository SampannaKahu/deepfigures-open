# /home/sampanna/workspace/bdts2/deepfigures-results/pmctable_arxiv_combined_2019_11_29_04.12

import os
from pprint import pprint
import tensorflow as tf

PATH = '/home/sampanna/workspace/bdts2/deepfigures-results/pmctable_arxiv_combined_2019_11_29_04.12/save.ckpt-550000'
tf_path = os.path.abspath(PATH)  # Path to our TensorFlow checkpoint
tf_vars = tf.train.list_variables(tf_path)
pprint(tf_vars)

import torch.nn as nn
import torchvision

resnet101 = torchvision.models.resnet101()
