import os
import json
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from typing import Dict, List
from deepfigures.extraction.datamodels import BoxClass
from deepfigures.extraction import figure_utils

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


def rect_to_box_class(rect: Dict) -> BoxClass:
    return BoxClass(x1=rect['x1'], y1=rect['y1'], x2=rect['x2'], y2=rect['y2'])


def plot_bbs(image_path: str, bbs: List[BoxClass], save_fig=False, bb_color='g'):
    if not bbs:
        return

    im = np.array(Image.open(image_path), dtype=np.uint8)
    fig, ax = plt.subplots(1)
    ax.imshow(im)

    for bb in bbs:
        x1, y1, x2, y2 = bb.x1, bb.y1, bb.x2, bb.y2
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=bb_color, facecolor='none')
        ax.add_patch(rect)

    if save_fig:
        plt.savefig(image_path.split('.png')[0] + '_bb.png')
    plt.show()


if __name__ == "__main__":
    path_to_figure_boundaries_with_hidden_detection_file = "/home/sampanna/workspace/bdts2/deepfigures-results/model_checkpoints/377266_arxiv_2020-06-02_22-48-45/figure_boundaries_hidden_set_501101.json"
    dataset_dir = '/home/sampanna/workspace/bdts2/deepfigures-results/gold_standard_dataset'
    annos = json.load(open(path_to_figure_boundaries_with_hidden_detection_file))
    for anno in annos:
        true_boxes = [rect_to_box_class(rect) for rect in anno['rects']]
        pred_boxes = [rect_to_box_class(rect) for rect in anno['hidden_set_rects']]
        (true_indices, pred_indices) = figure_utils.pair_boxes(true_boxes, pred_boxes)
        if len(true_boxes) > 2 and len(pred_boxes) > 2:
            # plot_bbs(os.path.join(dataset_dir, 'images', anno['image_path']), bbs=true_boxes, bb_color='g')
            # plot_bbs(os.path.join(dataset_dir, 'images', anno['image_path']), bbs=pred_boxes, bb_color='r')
            print("Image: {}".format(anno['image_path']))
            print("True boxes: {}".format(true_boxes))
            print("Pred boxes: {}".format(pred_boxes))
            print("True indices: {}".format(true_indices))
            print("Pred indices: {}".format(pred_indices))
