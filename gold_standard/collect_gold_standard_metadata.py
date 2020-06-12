import os
import json
import logging
import shutil

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


def get_gold_standard_metadata(gold_standard_dir: str, metadata_dir: str) -> None:
    figure_boundaries = json.load(open(os.path.join(gold_standard_dir, 'figure_boundaries.json')))
    metadata_paths = list(
        set([os.path.join(metadata_dir, anno['image_path'].split('.')[0] + '.json') for anno in figure_boundaries]))
    gold_standard_metadata_dir = os.path.join(gold_standard_dir, 'metadata')
    os.makedirs(gold_standard_metadata_dir, exist_ok=True)
    for metadata_path in metadata_paths:
        shutil.copyfile(metadata_path, os.path.join(gold_standard_metadata_dir, os.path.basename(metadata_path)))


if __name__ == "__main__":
    gold_standard_dir = '/home/sampanna/workspace/bdts2/deepfigures-results/gold_standard_dataset'
    mit_etd_metadata_dir = '/home/sampanna/workspace/bdts2/deepfigures-results/mit_etd_metadata'
    get_gold_standard_metadata(gold_standard_dir, mit_etd_metadata_dir)
