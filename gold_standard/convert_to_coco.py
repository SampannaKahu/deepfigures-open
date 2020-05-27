import os
import csv
import json
import logging

from PIL import Image

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

dataset_dir = '/home/sampanna/gold_standard_generation/final_gold_standard_dataset'
annotations_csv_file = '/home/sampanna/gold_standard_generation/final_gold_standard_dataset/annotations.csv'
images_dir = os.path.join(dataset_dir, 'images')


def build_annotation(x1: int, y1: int, height: int, width: int, annotation_id: int, image_id: int, category_id: int):
    """
    Sample annotation:
    {
        "segmentation": [
          [510.66,423.01,511.72,420.03,510.45,416,510.45,423.01]
        ],
        "area": 702.1057499999998,
        "iscrowd": 0,
        "image_id": 289343,
        "bbox": [
          473.07,
          395.93,
          38.65,
          28.67
        ],
        "category_id": 18,
        "id": 1768
    }
    """
    # x1, y1, x2, y2 = bb[1].item(), bb[3].item(), bb[2].item(), bb[4].item()
    x2 = x1 + width
    y2 = y1 + height
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2]],
        "area": width * height,
        "bbox": [x1, y1, width, height],
        "iscrowd": 0
    }


def build_image(image_path: str, image_id: int, height: int, width: int):
    """
    Sample image:
    {
        "license": 4,
        "file_name": "000000397133.jpg",
        "coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg",
        "height": 427,
        "width": 640,
        "date_captured": "2013-11-14 17:02:52",
        "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",
        "id": 397133
    }
    """
    return {
        "license": 2,  # TODO: Confirm this.
        "file_name": os.path.basename(image_path),
        "coco_url": "",
        "height": height,
        "width": width,
        "date_captured": "2020-05-20 01:00:00",
        "flickr_url": "",
        "id": image_id
    }


image_name_to_row_list_dict = {}

with open(annotations_csv_file) as fp:
    reader = csv.reader(fp)
    header_done = False
    for row in reader:
        if not header_done:
            header_done = True
            continue
        image_name = row[0]
        row_list = image_name_to_row_list_dict.get(image_name, [])
        rect_dict = json.loads(row[5])
        if rect_dict:
            row_list.append(rect_dict)
        image_name_to_row_list_dict[image_name] = row_list

dataset = json.load(open('/home/sampanna/deepfigures-open/hpc/post_process/coco_dataset_template.json'))
image_id = 1
annotation_id = 1
for image_name in image_name_to_row_list_dict:
    image_path = os.path.join(images_dir, image_name)
    img = Image.open(image_path)
    image_json = build_image(image_path=image_name, image_id=image_id, height=img.size[0], width=img.size[1])
    dataset['images'].append(image_json)
    row_list = image_name_to_row_list_dict[image_name]
    for row in row_list:
        annotation_json = build_annotation(x1=row['x'], y1=row['y'], width=row['width'], height=row['height'],
                                           annotation_id=annotation_id, image_id=image_id, category_id=1)
        dataset['annotations'].append(annotation_json)
        annotation_id = annotation_id + 1
    image_id = image_id + 1

json.dump(dataset, open(os.path.join(dataset_dir, 'annotations.json'), mode='w'), indent=2)
