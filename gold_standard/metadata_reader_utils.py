import os
import util
import json
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


def get_year_for_image_name(image_name: str, gold_standard_dir: str) -> int:
    handle = util.image_name_to_handle(image_name)
    metadata = json.load(open(os.path.join(gold_standard_dir, 'metadata.json')))
    issued_str = metadata[handle]['dc.date.issued'][0]
    return int(issued_str.split('-')[0])


if __name__ == "__main__":
    year = get_year_for_image_name('1721_1_44468.pdf-dpi100-page0002.png',
                                   '/home/sampanna/workspace/bdts2/deepfigures-results/gold_standard_dataset')
    print(year)
