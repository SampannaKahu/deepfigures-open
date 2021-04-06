import os
import json
import logging
from pprint import pformat

from scripts.hidden_set_eval import rect_to_box_class
from deepfigures.extraction import figure_utils

# Logging config
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

# Some constants
TOTAL_IMAGES_PROCESSED = "total_images_processed"
TOTAL_ANNOS_1 = "total_annos_1"
TOTAL_ANNOS_2 = "total_annos_2"
TOTAL_MATCHED_1 = "total_matched_1"
TOTAL_MATCHED_2 = "total_matched_2"
TOTAL_UNMATCHED_1 = "total_unmatched_1"
TOTAL_UNMATCHED_2 = "total_unmatched_2"
TOTAL_IOU_PASSED_1 = "total_iou_passed_1"
TOTAL_IOU_PASSED_2 = "total_iou_passed_2"
TOTAL_IOU_FAILED_1 = "total_iou_failed_1"
TOTAL_IOU_FAILED_2 = "total_iou_failed_2"

gold_standard_dir = '/home/sampanna/workspace/bdts2/deepfigures-results/gold_standard_dataset'

figure_boundaries_1_path = os.path.join(gold_standard_dir, 'figure_boundaries.json')
figure_boundaries_2_path = os.path.join(gold_standard_dir, 'figure_boundaries_second.json')

with open(figure_boundaries_1_path) as fp:
    figures_boundaries_1_dict = {anno['image_path']: anno for anno in json.load(fp)}

with open(figure_boundaries_2_path) as fp:
    figures_boundaries_2_dict = {anno['image_path']: anno for anno in json.load(fp)}

set_1 = set(figures_boundaries_1_dict.keys())

counts = {
    TOTAL_IMAGES_PROCESSED: 0,
    TOTAL_ANNOS_1: 0,
    TOTAL_ANNOS_2: 0,
    TOTAL_MATCHED_1: 0,
    TOTAL_MATCHED_2: 0,
    TOTAL_UNMATCHED_1: 0,
    TOTAL_UNMATCHED_2: 0,
    TOTAL_IOU_PASSED_1: 0,
    TOTAL_IOU_PASSED_2: 0,
    TOTAL_IOU_FAILED_1: 0,
    TOTAL_IOU_FAILED_2: 0
}

for image_path in set_1:
    anno_1 = figures_boundaries_1_dict[image_path]
    anno_2 = figures_boundaries_2_dict[image_path]

    if len(anno_1['rects']) != len(anno_2['rects']):
        logger.info("Different number of boxes found.")

    # Transform annos to BoxClass
    boxes_1 = [rect_to_box_class(rect) for rect in anno_1['rects']]
    boxes_2 = [rect_to_box_class(rect) for rect in anno_2['rects']]

    # Hungarian matching
    (indices_1, indices_2) = figure_utils.pair_boxes(boxes_1, boxes_2)

    if len(indices_1) != len(boxes_1) or len(indices_2) != len(boxes_2):
        logger.info("Unmatched boxes found.")

    # Find unmatched boxes
    unmatched_boxes_1 = [box for idx, box in enumerate(boxes_1) if idx not in indices_1]
    unmatched_boxes_2 = [box for idx, box in enumerate(boxes_2) if idx not in indices_2]

    # Initialize lists for the loop
    iou_passed_boxes_1 = []
    iou_passed_boxes_2 = []
    iou_failed_boxes_1 = []
    iou_failed_boxes_2 = []

    # Separate/sort the matched boxes based on IOU threshold
    for (index_1, index_2) in zip(indices_1, indices_2):
        matched_box_1 = boxes_1[index_1]
        matched_box_2 = boxes_2[index_2]
        iou = matched_box_1.iou(matched_box_2)
        # logger.info(iou)
        if iou > 0.8:
            iou_passed_boxes_1.append(matched_box_1)
            iou_passed_boxes_2.append(matched_box_2)
        else:
            iou_failed_boxes_1.append(matched_box_1)
            iou_failed_boxes_2.append(matched_box_2)

    counts[TOTAL_IMAGES_PROCESSED] += 1
    counts[TOTAL_ANNOS_1] += len(anno_1)
    counts[TOTAL_ANNOS_2] += len(anno_2)
    counts[TOTAL_MATCHED_1] += len(indices_1)
    counts[TOTAL_MATCHED_2] += len(indices_2)
    counts[TOTAL_UNMATCHED_1] += len(unmatched_boxes_1)
    counts[TOTAL_UNMATCHED_2] += len(unmatched_boxes_2)
    counts[TOTAL_IOU_PASSED_1] += len(iou_passed_boxes_1)
    counts[TOTAL_IOU_PASSED_2] += len(iou_passed_boxes_2)
    counts[TOTAL_IOU_FAILED_1] += len(iou_failed_boxes_1)
    counts[TOTAL_IOU_FAILED_2] += len(iou_failed_boxes_2)

logger.info(pformat(counts, indent=2))
