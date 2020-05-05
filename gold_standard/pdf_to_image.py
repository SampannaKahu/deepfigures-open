import os
import glob
import json
import util
import shutil
import logging
from deepfigures.utils import settings_utils
from deepfigures import settings

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

pdf_renderer = settings_utils.import_setting(settings.DEEPFIGURES_PDF_RENDERER)()

output_dir = '/home/sampanna/mit_etd_data/page_images'

# input_pattern = '/home/sampanna/mit_etd_data/data/etds/*.pdf'
# pdf_paths = list(map(os.path.abspath, glob.glob(input_pattern)))
pdf_paths = ['/home/sampanna/mit_etd_data/data/etds/' + util.handle_to_pdffilename(handle) for handle in
             json.load(open('data/sampled_etds.json'))]
logger.info("Total paths obtained: {path_count}".format(path_count=len(pdf_paths)))

for pdf_path in pdf_paths:
    logger.info(" ")
    logger.info("----------------------------------------------------------------------------------------")
    logger.info("Converting {pdf}.".format(pdf=pdf_path))
    image_paths = pdf_renderer.render(
        pdf_path=pdf_path,
        dpi=100,
        max_pages=500,
        output_dir=output_dir,
        use_cache=False
    )
    for image_path in image_paths:
        dest_image_path = os.path.join(output_dir, os.path.basename(image_path))
        logger.info("Moving {src} to {dest}".format(src=image_path, dest=dest_image_path))
        os.rename(image_path, dest_image_path)
        logger.info("Moved {src} to {dest}".format(src=image_path, dest=dest_image_path))
    if image_paths and os.path.exists(os.path.dirname(os.path.dirname(os.path.dirname(image_paths[0])))):
        dir_to_delete = os.path.dirname(os.path.dirname(os.path.dirname(image_paths[0])))
        shutil.rmtree(dir_to_delete)
