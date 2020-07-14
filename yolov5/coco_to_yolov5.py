import os
import json
import shutil
import logging
from PIL import Image
from yolov5 import yolov5_anno_util

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


def convert_coco_to_yolov5(coco_images_dir: str, coco_json_anno_path: str, _output_dir: str, split: str):
    """
    Converts coco dataset format to the format required by the yolov5 model.
    :param coco_images_dir: the absolute path to the images directory of the coco dataset to be converted.
    :param coco_json_anno_path: the absolute path to the annotations of the coco dataset.
    :param _output_dir: the output directory to store the files. Two directories will be created in this (images, labels)
    :param split: This should be either train or val. A directory with this name will be created within the images and labels directories each.
    :return: None.
    """
    _output_images_dir = os.path.join(_output_dir, 'images', split)
    _output_labels_dir = os.path.join(_output_dir, 'labels', split)
    os.makedirs(_output_images_dir)
    os.makedirs(_output_labels_dir)
    figure_boundaries_validation_json = json.load(open(coco_json_anno_path))
    for anno in figure_boundaries_validation_json:
        shutil.move(os.path.join(coco_images_dir, anno['image_path']),
                    os.path.join(_output_images_dir, anno['image_path']))
        im = Image.open(os.path.join(_output_images_dir, anno['image_path']))
        width, height = im.size
        rects = [yolov5_anno_util.coco_dict_to_yolov5_line(rect, width, height) for rect in anno['rects']]
        if rects:
            anno_path = os.path.join(_output_labels_dir, anno['image_path'].split('.png')[0] + '.txt')
            with open(anno_path, mode='w') as anno_file:
                anno_file.write('\n'.join(rects) + '\n')


if __name__ == "__main__":
    coco_dataset_dir = '/work/cascades/sampanna/deepfigures-results/gold_standard_dataset'
    figure_boundaries_json_path = os.path.join(coco_dataset_dir, 'figure_boundaries_validation.json')
    output_dir = '/work/cascades/sampanna/yolov5/gold_standard'
    split = 'val'

    images_dir = os.path.join(coco_dataset_dir, 'images')

    convert_coco_to_yolov5(coco_images_dir=images_dir,
                           coco_json_anno_path=figure_boundaries_json_path,
                           _output_dir=output_dir,
                           split=split)
