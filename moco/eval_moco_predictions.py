import os
import json
import logging
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

ground_truth = json.load(open('/home/sampanna/Desktop/moco_some_results_10_Nov_2020/instances_val2017.json'))
predictions = json.load(open('/home/sampanna/Desktop/moco_some_results_10_Nov_2020/coco_instances_results.json'))
p = json.loads(json.dumps(ground_truth))
for i, ann in enumerate(predictions):
    predictions[i]['id'] = i
    predictions[i]['area'] = predictions[i]['bbox'][2] * predictions[i]['bbox'][3]
predictions = [p for p in predictions if p['score'] > 0.9]
p['annotations'] = predictions
predictions = p
json.dump(predictions,
          open('/home/sampanna/Desktop/moco_some_results_10_Nov_2020/coco_instances_results_full_coco.json', mode='w'))

coco_pred = COCO(
    annotation_file='/home/sampanna/Desktop/moco_some_results_10_Nov_2020/coco_instances_results_full_coco.json')

coco_gt = COCO(annotation_file='/home/sampanna/Desktop/moco_some_results_10_Nov_2020/instances_val2017.json')

coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_pred, iouType='segm')
coco_eval.params.iouThrs = np.linspace(.8, 0.8, 1, endpoint=True)
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
print('Done')
