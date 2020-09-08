import os
import random
import logging
from gold_standard.convert_VIA_to_coco import get_image_name_to_row_list_dict, create_annotations_for_image_names

random.seed(0)
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    dataset_dir = '/home/sampanna/workspace/bdts2/deepfigures-results/gold_standard_dataset'
    annotations_csv_file = '/home/sampanna/workspace/bdts2/deepfigures-results/gold_standard_dataset/annotations.csv'
    images_dir = os.path.join(dataset_dir, 'images')

    image_name_to_row_list_dict = get_image_name_to_row_list_dict(annotations_csv_file)

    image_names = list(image_name_to_row_list_dict.keys())
    random.shuffle(image_names)

    train_image_names = image_names[:int(len(image_names) * 0.8)]
    create_annotations_for_image_names(image_names=train_image_names,
                                       _image_name_to_row_list_dict=image_name_to_row_list_dict,
                                       _images_dir=os.path.join(dataset_dir, 'images'),
                                       _output_json_path=os.path.join(dataset_dir, 'train_annotations.json'))

    val_image_names = image_names[int(len(image_names) * 0.8):]
    create_annotations_for_image_names(image_names=val_image_names,
                                       _image_name_to_row_list_dict=image_name_to_row_list_dict,
                                       _images_dir=os.path.join(dataset_dir, 'images'),
                                       _output_json_path=os.path.join(dataset_dir, 'val_annotations.json'))
