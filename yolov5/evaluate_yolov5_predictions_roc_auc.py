import os
import logging
import numpy as np
from yolov5.evaluate_yolov5_predictions import evaluate_yolov5_predictions_for_set

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

gold_standard_dir = "/work/cascades/sampanna/deepfigures-results/gold_standard_dataset"
predictions_output_dir = "/work/cascades/sampanna/yolov5/test_conf_thresh_inference_outputs/0"
metadata_json_path = os.path.join(gold_standard_dir, 'metadata.json')

conf_thres_list = list(np.arange(start=0, stop=1, step=0.01))
for conf_thres in conf_thres_list:
    suffix = '_conf_thres_%f' % conf_thres if conf_thres else ''

    evaluate_yolov5_predictions_for_set(gold_standard_dir,
                                        predictions_output_dir,
                                        metadata_json_path,
                                        os.path.join(gold_standard_dir, 'figure_boundaries.json'),
                                        suffix)
