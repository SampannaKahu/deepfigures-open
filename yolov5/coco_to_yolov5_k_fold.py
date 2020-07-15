import os
import copy
import json
import shutil
import logging
from typing import Tuple, List
from yolov5.coco_to_yolov5 import convert_coco_to_yolov5
from multiprocessing import Pool, cpu_count
from functools import partial

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


def get_annos_for_fold(fold: int, anno_folds: list) -> Tuple[List, List]:
    _val_annos = copy.deepcopy(anno_folds[fold])
    _train_annos = []
    for f, _l in enumerate(anno_folds):
        if f is not fold:
            _train_annos = _train_annos + copy.deepcopy(_l)
    return _train_annos, _val_annos


def process_fold(fold, anno_folds, coco_images_dir, output_dir):
    fold_dir = os.path.join(output_dir, str(fold))
    os.makedirs(fold_dir)
    train_annos, val_annos = get_annos_for_fold(fold, anno_folds)
    train_anno_path = os.path.join(fold_dir, 'figure_boundaries_train.json')
    json.dump(train_annos, open(train_anno_path, mode='w'))
    convert_coco_to_yolov5(coco_images_dir, train_anno_path, fold_dir, 'train')
    val_anno_path = os.path.join(fold_dir, 'figure_boundaries_val.json')
    json.dump(val_annos, open(val_anno_path, mode='w'))
    convert_coco_to_yolov5(coco_images_dir, val_anno_path, fold_dir, 'val')


def coco_to_yolov5_k_fold(k: int, output_dir: str, coco_images_dir: str, coco_anno_path: str):
    coco_annos = json.load(open(coco_anno_path))
    anno_folds = [coco_annos[i * k:(i + 1) * k] for i in range((len(coco_annos) + k - 1) // k)]
    os.makedirs(output_dir)
    with Pool(cpu_count()) as pool:
        pool.map(
            partial(process_fold, anno_folds=anno_folds, coco_images_dir=coco_images_dir, output_dir=output_dir),
            list(range(k))
        )


if __name__ == "__main__":
    K = 10
    output_dir = '/work/cascades/sampanna/yolov5/gold_standard_k_fold'
    _coco_images_dir = '/work/cascades/sampanna/deepfigures-results/gold_standard_dataset/images'
    _coco_anno_path = '/work/cascades/sampanna/deepfigures-results/gold_standard_dataset/annotations.json'
    shutil.rmtree(output_dir)
    coco_to_yolov5_k_fold(K, "./output", _coco_images_dir, _coco_anno_path)
