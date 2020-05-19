import os
import json
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

figure_boundaries_save_path = '/home/sampanna/deepfigures-results/arxiv_coco_dataset/figure_boundaries.json'
coco_annotation_file = '/home/sampanna/deepfigures-results/arxiv_coco_dataset/annotations.json'
coco_annos = json.load(open(coco_annotation_file))

image_id_to_file_name_map = {image['id']: image['file_name'] for image in coco_annos['images']}
image_id_to_figure_json_map = dict()
for figure in coco_annos['annotations']:
    figure_json = {
        "image_path": image_id_to_file_name_map[figure['image_id']],
        "rects": [
            {
                "x1": figure['segmentation'][0][0],
                "x2": figure['segmentation'][0][1],
                "y1": figure['segmentation'][0][2],
                "y2": figure['segmentation'][0][7]
            }
        ]
    }
    if figure['image_id'] in image_id_to_figure_json_map:
        image_id_to_figure_json_map[figure['image_id']]['rects'].append(figure_json["rects"][0])
    else:
        image_id_to_figure_json_map[figure['image_id']] = figure_json

json.dump(list(image_id_to_figure_json_map.values()), open(figure_boundaries_save_path, mode='w'))
