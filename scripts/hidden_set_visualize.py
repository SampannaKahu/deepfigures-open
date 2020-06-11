import os
import json
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch

from PIL import Image
from typing import Dict, List
from deepfigures.extraction.datamodels import BoxClass
from deepfigures.extraction import figure_utils

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


def rect_to_box_class(rect: Dict) -> BoxClass:
    return BoxClass(x1=rect['x1'], y1=rect['y1'], x2=rect['x2'], y2=rect['y2'])


def plot_bbs(image_path: str, bbs: List[BoxClass], save_fig=False, bb_color='g', fig=None, ax=None):
    if not bbs:
        return

    if not fig or not ax:
        im = np.array(Image.open(image_path), dtype=np.uint8)
        fig, ax = plt.subplots(1)
        ax.imshow(im)
        plt.title(os.path.basename(image_path))

    for bb in bbs:
        x1, y1, x2, y2 = bb.x1, bb.y1, bb.x2, bb.y2
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=bb_color, facecolor='none')
        ax.add_patch(rect)

    if save_fig:
        plt.savefig(image_path.split('.png')[0] + '_bb.png')
    plt.show()
    return (fig, ax)


def plot_line(x1: int, y1: int, x2: int, y2: int, fig=None, ax=None):
    if not fig or not ax:
        fig, ax = plt.subplots(1)

    _x1 = [x1, x2]
    _y1 = [y1, y2]
    plt.plot(_x1, _y1, 'b', marker='o')
    plt.show()
    return (fig, ax)


def compute_page_iou(pred_boxes: List[BoxClass], true_boxes: List[BoxClass],
                     pred_indices: List[int], true_indices: List[int]) -> List[float]:
    ious = []
    for (true_idx, pred_idx) in zip(true_indices, pred_indices):
        ious = ious.append(true_boxes[true_idx].iou(pred_boxes[pred_idx]))

    return ious + [0.0] * (max(len(pred_boxes), len(true_boxes)) - max(len(pred_indices), len(true_indices)))


if __name__ == "__main__":
    path_to_figure_boundaries_with_hidden_detection_file = "/home/sampanna/workspace/bdts2/deepfigures-results/model_checkpoints/377266_arxiv_2020-06-02_22-48-45/figure_boundaries_hidden_set_501101.json"
    dataset_dir = '/home/sampanna/workspace/bdts2/deepfigures-results/gold_standard_dataset'

    annos = json.load(open(path_to_figure_boundaries_with_hidden_detection_file))
    counter = 0
    for anno in annos:
        true_boxes = [rect_to_box_class(rect) for rect in anno['rects']]
        pred_boxes = [rect_to_box_class(rect) for rect in anno['hidden_set_rects']]
        (true_indices, pred_indices) = figure_utils.pair_boxes(true_boxes, pred_boxes)
        if len(true_boxes) > 2 and len(pred_boxes) > 2:
            counter = counter + 1
            (fig, ax) = plot_bbs(os.path.join(dataset_dir, 'images', anno['image_path']), bbs=true_boxes, bb_color='g')
            (fig, ax) = plot_bbs(os.path.join(dataset_dir, 'images', anno['image_path']), bbs=pred_boxes, bb_color='r',
                                 fig=fig, ax=ax)
            plt.legend(handles=[
                Patch(color='r', label='Predicted'),
                Patch(color='g', label='Ground truth'),
                Patch(color='b', label='Box correspondence')
            ])
            for (true_idx, pred_idx) in zip(true_indices, pred_indices):
                (fig, ax) = plot_line(true_boxes[true_idx].x1, true_boxes[true_idx].y1,
                                      pred_boxes[pred_idx].x1, pred_boxes[pred_idx].y1,
                                      fig=fig, ax=ax)
            print("Image: {}".format(anno['image_path']))
            print("True boxes: {}".format(true_boxes))
            print("Pred boxes: {}".format(pred_boxes))
            print("True indices: {}".format(true_indices))
            print("Pred indices: {}".format(pred_indices))
            if counter >= 10:
                break
