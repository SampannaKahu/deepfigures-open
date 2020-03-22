# /home/sampanna/workspace/bdts2/deepfigures-results/pmctable_arxiv_combined_2019_11_29_04.12

import os, json
from pprint import pprint
import tensorflow as tf

PATH = '/home/sampanna/workspace/bdts2/deepfigures-results/pmctable_arxiv_combined_2019_11_29_04.12/save.ckpt-550000'
tf_path = os.path.abspath(PATH)  # Path to our TensorFlow checkpoint
init_vars = tf.train.list_variables(tf_path)
pprint(init_vars)
# json.dump(tf_vars, open('vars.json', mode='w'), indent=2)

# tf_vars = []
# for name, shape in init_vars:
#     # print("Loading TF weight {} with shape {}".format(name, shape))
#     array = tf.train.load_variable(tf_path, name)
#     tf_vars.append((name, array.squeeze()))

v = tf.train.load_variable(tf_path, 'resnet_v1_101/block1/unit_1/bottleneck_v1/conv1/BatchNorm/beta/RMSProp')
# print(v)

import torch.nn as nn
import torchvision

#
resnet101 = torchvision.models.resnet101()
print(resnet101)
a = 2
# for param in resnet101.parameters():
#     param
#     print(param)

