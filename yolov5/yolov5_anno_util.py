import os
import logging
import torch
from typing import Tuple, Dict

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


def yolov5_line_to_coco_dict(yolov5_line: str, image_width, image_height) -> Dict:
    """
    :param yolov5_box_line: one line from the yolov5 annotation file. Should contain the class ID as the first word of the line.
    :return: COCO dict
    """
    parts = yolov5_line.strip().split()[1:]
    assert len(parts) == 4, "Expected 4 parts. Got " + str(len(parts))
    x_centre, y_centre, box_width, box_height = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
    x1, y1, x2, y2 = yolov5_anno_to_coco_anno(x_centre, y_centre, box_width, box_height, image_width, image_height)
    return {
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2
    }


def coco_dict_to_yolov5_line(coco_dict: Dict, image_width, image_height, category: str = '0') -> str:
    x_centre, y_centre, box_width, box_height = coco_anno_to_yolov5_anno(coco_dict['x1'], coco_dict['y1'],
                                                                         coco_dict['x2'], coco_dict['y2'], image_width,
                                                                         image_height)
    return ' '.join([
        category,  # class id
        str(x_centre),  # xc
        str(y_centre),  # yc
        str(box_width),  # bb_width
        str(box_height)  # bb_height
    ])


def yolov5_anno_to_coco_anno(x_centre, y_centre, box_width, box_height, image_width, image_height) -> Tuple[
    float, float, float, float]:
    x_centre, y_centre, box_width, box_height, image_width, image_height = float(x_centre), float(y_centre), float(
        box_width), float(box_height), image_width, float(image_height)
    x1 = (x_centre - (box_width / 2)) * image_width
    y1 = (y_centre - (box_height / 2)) * image_height
    x2 = (x_centre + (box_width / 2)) * image_width
    y2 = (y_centre + (box_height / 2)) * image_height
    return x1, y1, x2, y2


def coco_anno_to_yolov5_anno(x1, y1, x2, y2, image_width, image_height) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2, image_width, image_height = float(x1), float(y1), float(x2), float(y2), float(image_width), float(
        image_height)
    x_centre = (x1 + x2) / (2 * image_width)
    y_centre = (y1 + y2) / (2 * image_height)
    box_width = (x2 - x1) / image_width
    box_height = (y2 - y1) / image_height
    return x_centre, y_centre, box_width, box_height


def bb_tensor_to_coco_dict(bb_tensor: torch.tensor) -> Dict:
    assert bb_tensor.size()[0] == 1
    assert bb_tensor.size()[1] == 5
    return {
        'x1': bb_tensor[1].item(),
        'y1': bb_tensor[3].item(),
        'x2': bb_tensor[2].item(),
        'y2': bb_tensor[4].item(),
    }
