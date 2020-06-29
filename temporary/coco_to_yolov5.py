import os
import json
import shutil
import logging
from PIL import Image

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

output_dir = '/work/cascades/sampanna/yolov5/gold_standard'
output_images_dir = os.path.join(output_dir, 'images')
output_labels_dir = os.path.join(output_dir, 'labels')

gold_standard_dir = '/work/cascades/sampanna/deepfigures-results/gold_standard_dataset'
images_dir = os.path.join(gold_standard_dir, 'images')
figure_boundaries_validation = os.path.join(gold_standard_dir, 'figure_boundaries_validation.json')
figure_boundaries_validation_json = json.load(open(figure_boundaries_validation))

for anno in figure_boundaries_validation_json:
    shutil.move(os.path.join(images_dir, anno['image_path']), os.path.join(output_images_dir, anno['image_path']))
    im = Image.open(os.path.join(output_images_dir, anno['image_path']))
    width, height = im.size
    rects = [' '.join([
        '0',  # class id
        str((rect['x2'] + rect['x1']) / (2 * width)),  # xc
        str((rect['y2'] + rect['y1']) / (2 * height)),  # yc
        str((rect['x2'] - rect['x1']) / width),  # bb_width
        str((rect['y2'] - rect['y1']) / height)  # bb_height
    ]) for rect in anno['rects']]
    if rects:
        anno_path = os.path.join(output_labels_dir, anno['image_path'].split('.png')[0] + '.txt')
        with open(anno_path, mode='w') as anno_file:
            anno_file.write('\n'.join(rects) + '\n')
