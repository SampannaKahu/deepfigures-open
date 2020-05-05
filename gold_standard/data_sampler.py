import os
import json
import random
import logging
import util
import shutil
from deepfigures.utils import settings_utils
from deepfigures import settings

random.seed(42)
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)
pdf_renderer = settings_utils.import_setting(settings.DEEPFIGURES_PDF_RENDERER)()

SAMPLED_ETD_PATH = 'data/sampled_etds.json'
ALL_METADATA_PATH = 'data/all_metadata.json'
IMAGE_OUTPUT_DIR = '/home/sampanna/mit_etd_data/page_images'


def pdf_to_images(handle: str) -> bool:
    """
    Converts PDFs to page-wise images.
    :param handle: the handle of the etd to convert to images.
    :return: true if this pdf was successfully converted to images.
    """
    pdf_path = '/home/sampanna/mit_etd_data/data/etds/' + util.handle_to_pdffilename(handle)
    logger.info(" ")
    logger.info("----------------------------------------------------------------------------------------")
    logger.info("Converting {pdf}.".format(pdf=pdf_path))
    image_paths = pdf_renderer.render(
        pdf_path=pdf_path,
        dpi=100,
        max_pages=500,
        output_dir=IMAGE_OUTPUT_DIR,
        use_cache=False
    )
    for image_path in image_paths:
        dest_image_path = os.path.join(IMAGE_OUTPUT_DIR, os.path.basename(image_path))
        logger.info("Moving {src} to {dest}".format(src=image_path, dest=dest_image_path))
        os.rename(image_path, dest_image_path)
        logger.info("Moved {src} to {dest}".format(src=image_path, dest=dest_image_path))
    if image_paths and os.path.exists(os.path.dirname(os.path.dirname(os.path.dirname(image_paths[0])))):
        dir_to_delete = os.path.dirname(os.path.dirname(os.path.dirname(image_paths[0])))
        shutil.rmtree(dir_to_delete)
        return True
    return False


def get_year_issued(all_metadata: dict, handle: str) -> int:
    issued_str = all_metadata[handle]['dc.date.issued'][0]
    return int(issued_str.split('-')[0])


with open(ALL_METADATA_PATH) as fp:
    all_metadata = json.load(fp)

with open('mit_depts_with_subcommunities.json') as fp:
    mit_index = json.load(fp)

with open(SAMPLED_ETD_PATH) as fp:
    sampled_etds = json.load(fp)

sampled_etds_set = set(sampled_etds)
issue_date_upper = 1990  # inclusive
issue_date_lower = 1890  # inclusive

for dept in mit_index:
    for sub_comm in dept['sub_communities']:
        sub_comm_handle = sub_comm['handle']
        # print(sub_comm_handle)
        sub_comm_json_path = 'data/' + util.handle_to_filename(sub_comm_handle)
        if not os.path.isfile(sub_comm_json_path):
            logger.warning("Skipping sub comm: {} since no file found".format(sub_comm_handle))
            continue
        with open(sub_comm_json_path) as fp:
            sub_comm_etd_handles = [etd_obj for etd_obj in json.load(fp) if get_year_issued(all_metadata, etd_obj[
                'handle']) <= issue_date_upper and get_year_issued(all_metadata, etd_obj['handle']) >= issue_date_lower]
            if not sub_comm_etd_handles:
                logger.warning("Skipping sub comm: {} since no applicable etd found".format(sub_comm_handle))
                continue

            unsampled_etd_handles = [etd_handle for etd_handle in sub_comm_etd_handles if
                                     etd_handle['handle'] not in sampled_etds_set]
            if not unsampled_etd_handles:
                logger.warning(
                    "Skipping sub comm: {} since all etds have already been sampled".format(sub_comm_handle))
                continue

            random.shuffle(unsampled_etd_handles)  # in-place shuffle.
            for unsample_etd_handle in unsampled_etd_handles:
                _handle_str = unsample_etd_handle['handle']
                logger.info('Attempting to convert etd {} to images.'.format(_handle_str))
                if pdf_to_images(_handle_str):
                    logger.info('Successfully converted etd {} to images.'.format(_handle_str))
                    sampled_etds.append(_handle_str)
                    json.dump(sampled_etds, open(SAMPLED_ETD_PATH, mode='w'))
                    break
                else:
                    logger.warning("Failed to convert etd {} to images. Skipping".format(_handle_str))

print(len(sampled_etds))
print(sampled_etds)

json.dump(sampled_etds, open(SAMPLED_ETD_PATH, mode='w'))
