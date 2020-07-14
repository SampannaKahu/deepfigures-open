import os
import json
import logging

from PIL import Image
from yolov5.yolov5_anno_util import yolov5_line_to_coco_dict
from scripts.hidden_set_eval import compute_metrics_for_annos

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


# gold_standard_dir = '/home/sampanna/workspace/bdts2/deepfigures-results/gold_standard_dataset'
# run_for_set = None  # can be either 'validation', 'testing' or None. If None, all annos will be used.
# predictions_output_dir = '/home/sampanna/Desktop/yolov5_72_hr_nr_training_results/output'

# annos = json.load(open(os.path.join(gold_standard_dir, 'figure_boundaries.json')))
# for anno in annos:
#     image_path = anno['image_path']
#     if os.path.isfile(os.path.join(predictions_output_dir, image_path.split(".png")[0] + '.txt')):
#         width, height = Image.open(os.path.join(gold_standard_dir, 'images', image_path)).size
#         with open(os.path.join(predictions_output_dir, image_path.split(".png")[0] + '.txt')) as pred_fp:
#             lines = pred_fp.readlines()
#             anno['hidden_set_rects'] = [yolov5_line_to_coco_dict(line, width, height) for line in lines]
#     else:
#         anno['hidden_set_rects'] = []
# # json.dump(annos, open(os.path.join(gold_standard_dir, 'figure_boundaries_yolov5.json'), mode='w'))
# compute_metrics_for_annos(annos, gold_standard_dir, run_for_set)


def evaluate_yolov5_predictions_for_set(_gold_standard_dir: str, _predictions_output_dir, run_for_set: str):
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
    compute_metrics_for_annos(annos, _gold_standard_dir, run_for_set)


if __name__ == "__main__":
    gold_standard_dir = '/home/sampanna/workspace/bdts2/deepfigures-results/gold_standard_dataset'
    predictions_output_dir = '/home/sampanna/Desktop/yolov5_72_hr_nr_training_results/output'
    run_for_set_list = [None, 'validation', 'testing']  # If None, all annos will be used.

    for run_for_set in run_for_set_list:
        print("Evaluating the predictions on the gold standard {}".format(run_for_set))
        evaluate_yolov5_predictions_for_set(gold_standard_dir, predictions_output_dir, run_for_set)
