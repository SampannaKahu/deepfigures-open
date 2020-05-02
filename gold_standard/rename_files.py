import os
import glob

import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


def rename(file):
    os.rename(file, os.path.join(os.path.dirname(file), os.path.basename(file).replace(".", "_", 1)))


etds_path = '/tmp'
files = glob.glob(os.path.join(etds_path, '*.pdf'))
for file in files:
    rename(file)
