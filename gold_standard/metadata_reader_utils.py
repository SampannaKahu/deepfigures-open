import os
import json
import logging
from gold_standard import util

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


def get_year_for_image_name(image_name: str, metadata: dict) -> int:
    handle = util.image_name_to_handle(image_name)
    return get_year_for_handle(handle, metadata)


def get_year_for_handle(handle: str, metadata: dict) -> int:
    issued_str = metadata[handle]['dc.date.issued'][0]
    return int(issued_str.split('-')[0])


if __name__ == "__main__":
    year = get_year_for_image_name('1721_1_44468.pdf-dpi100-page0002.png',
                                   '/home/sampanna/workspace/bdts2/deepfigures-results/gold_standard_dataset')
    print(year)
