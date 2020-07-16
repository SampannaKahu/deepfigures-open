import os
import json
import logging

from PIL import Image
from yolov5.yolov5_anno_util import yolov5_line_to_coco_dict
from scripts.hidden_set_eval import compute_metrics_for_anno_set

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


def evaluate_yolov5_predictions_for_set(_gold_standard_dir: str, _predictions_output_dir,
                                        metadata_json_path: str, filter_anno_path: str):
    annos = json.load(open(os.path.join(_gold_standard_dir, 'figure_boundaries.json')))
    for anno in annos:
        image_path = anno['image_path']
        if os.path.isfile(os.path.join(_predictions_output_dir, image_path.split(".png")[0] + '.txt')):
            width, height = Image.open(os.path.join(_gold_standard_dir, 'images', image_path)).size
            with open(os.path.join(_predictions_output_dir, image_path.split(".png")[0] + '.txt')) as pred_fp:
                lines = pred_fp.readlines()
                anno['hidden_set_rects'] = [yolov5_line_to_coco_dict(line, width, height) for line in lines]
        else:
            anno['hidden_set_rects'] = []
    # json.dump(annos, open(os.path.join(gold_standard_dir, 'figure_boundaries_yolov5.json'), mode='w'))
    filter_anno_list = json.load(open(filter_anno_path))
    compute_metrics_for_anno_set(annos, metadata_json_path, filter_anno_list)


if __name__ == "__main__":
    """
    Function:
    python -c 'from deepfigures.yolov5 import evaluate_yolov5_predictions; evaluate_yolov5_predictions.evaluate_yolov5_predictions_for_set("/work/cascades/sampanna/deepfigures-results/gold_standard_dataset", "/home/sampanna/yolov5_cross_val/0/yolov5/inference/output", "/work/cascades/sampanna/deepfigures-results/gold_standard_dataset/metadata.json", "/work/cascades/sampanna/yolov5/gold_standard_k_fold/0/figure_boundaries_val.json")'
    """

    gold_standard_dir = '/home/sampanna/workspace/bdts2/deepfigures-results/gold_standard_dataset'
    predictions_output_dir = '/home/sampanna/Desktop/yolov5_72_hr_nr_training_results/output'
    metadata_json_path = os.path.join(gold_standard_dir, 'metadata.json')
    run_for_set_list = [
        os.path.join(gold_standard_dir, 'figure_boundaries.json'),
        os.path.join(gold_standard_dir, 'figure_boundaries_validation.json'),
        os.path.join(gold_standard_dir, 'figure_boundaries_testing.json')
    ]

    for run_for_set in run_for_set_list:
        print("Evaluating the predictions on the gold standard {}".format(run_for_set))
        evaluate_yolov5_predictions_for_set(gold_standard_dir, predictions_output_dir, metadata_json_path, run_for_set)
