import os
import json
import logging
from typing import Dict, List, Tuple
from deepfigures.extraction.datamodels import BoxClass
from deepfigures.extraction import figure_utils
from gold_standard.metadata_reader_utils import get_year_for_image_name

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


def rect_to_box_class(rect: Dict) -> BoxClass:
    return BoxClass(x1=rect['x1'], y1=rect['y1'], x2=rect['x2'], y2=rect['y2'])


def compute_page_ious(pred_boxes: List[BoxClass], true_boxes: List[BoxClass], iou_thresh: float) -> Tuple[
    List[float], int, int, int]:
    """
    Reference: https://towardsdatascience.com/evaluating-performance-of-an-object-detection-model-137a349c517b
    :param pred_boxes:
    :param true_boxes:
    :return:
    """
    (true_indices, pred_indices) = figure_utils.pair_boxes(true_boxes, pred_boxes)
    tp, fp, fn = 0, 0, 0
    ious = []
    for (true_idx, pred_idx) in zip(true_indices, pred_indices):
        iou = true_boxes[true_idx].iou(pred_boxes[pred_idx])
        ious.append(iou)
        if iou >= iou_thresh:
            tp = tp + 1
        if iou < 0.8:
            fp = fp + 1
    fn = len(true_boxes) - len(true_indices)
    assert fn >= 0
    return ious + [0.0] * (
            max(len(pred_boxes), len(true_boxes)) - max(len(pred_indices), len(true_indices))), tp, fp, fn


def compute_mean_iou_for_annos(annos: list, iou_thresh: float) -> Tuple[float, int, int, int]:
    ious = []
    tp, fp, fn = 0, 0, 0
    for anno in annos:
        true_boxes = [rect_to_box_class(rect) for rect in anno['rects']]
        pred_boxes = [rect_to_box_class(rect) for rect in anno['hidden_set_rects']]
        _ious, _tp, _fp, _fn = compute_page_ious(pred_boxes, true_boxes, iou_thresh)
        ious = ious + _ious
        tp = tp + _tp
        fp = fp + _fp
        fn = fn + _fn
    if len(ious) == 0:
        return 0.0, tp, fp, fn
    return sum(ious) / len(ious), tp, fp, fn


def split_annos_year_wise(annos: list, gold_standard_dir: str) -> dict:
    metadata = json.load(open(os.path.join(gold_standard_dir, 'metadata.json')))
    year_to_anno_list_dict = {}
    for anno in annos:
        if 'etd_for_gold_standard_dataset_2' in anno['image_path']:
            year = 1111
        else:
            year = get_year_for_image_name(anno['image_path'], metadata)
        annos_for_year = year_to_anno_list_dict.get(year, [])
        annos_for_year.append(anno)
        year_to_anno_list_dict[year] = annos_for_year
    return year_to_anno_list_dict


def compute_precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    tp, fp, fn = float(tp), float(fp), float(fn)
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0

    if precision + recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0

    return precision, recall, f1


if __name__ == "__main__":
    gold_standard_dir = '/home/sampanna/workspace/bdts2/deepfigures-results/gold_standard_dataset'

    # path_to_figure_boundaries_with_hidden_detection_file = "/home/sampanna/workspace/bdts2/deepfigures-results/model_checkpoints/377266_arxiv_2020-06-02_22-48-45/figure_boundaries_hidden_set_501101.json"
    # path_to_figure_boundaries_with_hidden_detection_file = "/home/sampanna/workspace/bdts2/deepfigures-results/weights/figure_boundaries_hidden_set_2_500000.json"
    # path_to_figure_boundaries_with_hidden_detection_file = "/home/sampanna/workspace/bdts2/deepfigures-results/pmctable_arxiv_combined_2019_11_29_04.12/figure_boundaries_hidden_set_2_600000.json"
    # path_to_figure_boundaries_with_hidden_detection_file = "/home/sampanna/workspace/bdts2/deepfigures-results/pmctable_arxiv_combined_2019_11_29_09.10/figure_boundaries_hidden_set_600000.json"
    path_to_figure_boundaries_with_hidden_detection_file = "/home/sampanna/ir/deepfigures-results/model_checkpoints/377268_arxiv_2020-06-14_01-23-25/figure_boundaries_hidden_set_508701.json"
    annos = json.load(open(path_to_figure_boundaries_with_hidden_detection_file))
    annos_year_wise = split_annos_year_wise(annos, gold_standard_dir)
    annos_year_wise[0000] = annos
    for year, annos_for_year in annos_year_wise.items():
        _mean_iou, tp, fp, fn = compute_mean_iou_for_annos(annos=annos_for_year, iou_thresh=0.8)
        prec, rec, f1 = compute_precision_recall_f1(tp, fp, fn)
        print(year, (_mean_iou, tp, fp, fn, prec, rec, f1))
