import re
import numpy as np
import os
import tensorflow as tf
import torch

model = MyPyTorchGPT2()  # load the un-initialized PyTorch model we have created

# Retrieve weights from TF checkpoint
tf_path = os.path.abspath(gpt2_checkpoint_path)
init_vars = tf.train.list_variables(tf_path)
tf_vars = []
for name, shape in init_vars:
    print("Loading TF weight {} with shape {}".format(name, shape))
    array = tf.train.load_variable(tf_path, name)
    tf_vars.append((name, array.squeeze()))

# FOr each variable in the PyTorch model
for name, array in tf_vars:
    # skip the prefix ('model/') and split the path-like variable name in a list of sub-path
    name = name[6:].split('/')

    # Initiate the pointer from the main model class
    pointer = model

    # We iterate along the scopes and move our pointer accordingly
    for m_name in name:
        # we take special care of the `h0`, `h1`... paths and split them in `h` + the number
        if re.fullmatch(r'[A-Za-z]+\d+', m_name):
            l = re.split(r'(\d+)', m_name)
        else:
            l = [m_name]

        # Convert parameters final names to the PyTorch modules equivalent names
        if l[0] == 'w' or l[0] == 'g':
            pointer = getattr(pointer, 'weight')
        elif l[0] == 'b':
            pointer = getattr(pointer, 'bias')
        elif l[0] == 'wpe' or l[0] == 'wte':
            pointer = getattr(pointer, l[0])
            pointer = getattr(pointer, 'weight')
        else:
            pointer = getattr(pointer, l[0])

        # If we had a `hXX` name, let's access the sub-module with the right number
        if len(l) >= 2:
            num = int(l[1])
            pointer = pointer[num]
    try:
        assert pointer.shape == array.shape  # Catch error if the array shapes are not identical
    except AssertionError as e:
        e.args += (pointer.shape, array.shape)
        raise

    print("Initialize PyTorch weight {}".format(name))
    pointer.data = torch.from_numpy(array)
