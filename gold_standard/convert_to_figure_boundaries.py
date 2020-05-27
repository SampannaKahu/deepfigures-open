import os
import csv
import json
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


def via_to_figure_boundaries_rect(via_rect_dict: dict) -> dict:
    return {
        "x1": via_rect_dict['x'],
        "x2": via_rect_dict['x'] + via_rect_dict['width'],
        "y1": via_rect_dict['y'],
        "y2": via_rect_dict['x'] + via_rect_dict['height']
    }


def convert_gold_standard_to_figure_boundaries(annotations_csv_file: str, figure_boundaries_path: str):
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

    figure_boundaries = []
    for image_name in image_name_to_row_list_dict:
        rects = [via_to_figure_boundaries_rect(via_rect_dict) for via_rect_dict in
                 image_name_to_row_list_dict[image_name]]
        figure_boundaries.append({
            "image_path": image_name,
            "rects": rects
        })

    json.dump(figure_boundaries, open(figure_boundaries_path, mode='w'), indent=2)


if __name__ == "__main__":
    dataset_dir = '/home/sampanna/deepfigures-results/gold_standard/final_gold_standard_dataset'
    anno_csv_file = os.path.join(dataset_dir, 'annotations.csv')
    fig_bound_path = os.path.join(dataset_dir, 'figure_boundaries.json')
    convert_gold_standard_to_figure_boundaries(annotations_csv_file=anno_csv_file,
                                               figure_boundaries_path=fig_bound_path)
