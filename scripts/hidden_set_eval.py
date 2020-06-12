import os
import json
import logging
from typing import Dict, List
from deepfigures.extraction.datamodels import BoxClass
from deepfigures.extraction import figure_utils

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


def rect_to_box_class(rect: Dict) -> BoxClass:
    return BoxClass(x1=rect['x1'], y1=rect['y1'], x2=rect['x2'], y2=rect['y2'])


def compute_page_ious(pred_boxes: List[BoxClass], true_boxes: List[BoxClass]) -> List[float]:
    (true_indices, pred_indices) = figure_utils.pair_boxes(true_boxes, pred_boxes)
    ious = []
    for (true_idx, pred_idx) in zip(true_indices, pred_indices):
        ious.append(true_boxes[true_idx].iou(pred_boxes[pred_idx]))
    return ious + [0.0] * (max(len(pred_boxes), len(true_boxes)) - max(len(pred_indices), len(true_indices)))


if __name__ == "__main__":
    path_to_figure_boundaries_with_hidden_detection_file = "/home/sampanna/workspace/bdts2/deepfigures-results/model_checkpoints/377266_arxiv_2020-06-02_22-48-45/figure_boundaries_hidden_set_2_501101.json"
    # path_to_figure_boundaries_with_hidden_detection_file = "/home/sampanna/workspace/bdts2/deepfigures-results/weights/figure_boundaries_hidden_set_2_500000.json"
    # path_to_figure_boundaries_with_hidden_detection_file = "/home/sampanna/workspace/bdts2/deepfigures-results/pmctable_arxiv_combined_2019_11_29_04.12/figure_boundaries_hidden_set_2_600000.json"
    # path_to_figure_boundaries_with_hidden_detection_file = "/home/sampanna/workspace/bdts2/deepfigures-results/pmctable_arxiv_combined_2019_11_29_09.10/figure_boundaries_hidden_set_2_600000.json"
    annos = json.load(open(path_to_figure_boundaries_with_hidden_detection_file))
    ious = []
    for anno in annos:
        true_boxes = [rect_to_box_class(rect) for rect in anno['rects']]
        pred_boxes = [rect_to_box_class(rect) for rect in anno['hidden_set_rects']]
        ious = ious + compute_page_ious(pred_boxes, true_boxes)
    print(sum(ious) / len(ious))
