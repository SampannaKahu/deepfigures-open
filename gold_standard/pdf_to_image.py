import os
import glob
import util
import logging
import tempfile
from deepfigures.utils import settings_utils
from deepfigures import settings
from multiprocessing import Pool, cpu_count
from functools import partial

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


def convert_pdf_to_images(_pdf_path, _output_dir, _pdf_renderer, _temp_dir, max_pages=500):
    logger.info("Converting {pdf}.".format(pdf=_pdf_path))
    with tempfile.TemporaryDirectory(dir=_temp_dir) as td:
        image_paths = _pdf_renderer.render(
            pdf_path=_pdf_path,
            dpi=100,
            max_pages=max_pages,
            output_dir=td,
            use_cache=False
        )
        for image_path in image_paths:
            dest_image_path = os.path.join(_output_dir, os.path.basename(image_path))
            logger.info("Moving {src} to {dest}".format(src=image_path, dest=dest_image_path))
            os.rename(image_path, dest_image_path)
            logger.info("Moved {src} to {dest}".format(src=image_path, dest=dest_image_path))


def convert_pdf_paths_to_images(input_dir, output_dir, temp, cpu_count=cpu_count(), max_pages=500):
    pdf_renderer = settings_utils.import_setting(settings.DEEPFIGURES_PDF_RENDERER)()
    pdf_paths = glob.glob(os.path.join(input_dir, '*.pdf'))
    logger.info("Total paths obtained: {path_count}".format(path_count=len(pdf_paths)))
    with Pool(cpu_count) as pool:
        pool.map(
            partial(convert_pdf_to_images, _output_dir=output_dir, _pdf_renderer=pdf_renderer, _temp_dir=temp,
                    max_pages=max_pages),
            pdf_paths
        )


if __name__ == "__main__":
    temp = '/work/cascades/sampanna/moco/temp'
    output_dir = '/work/cascades/sampanna/moco/scanned_etd_dataset'
    input_dir = '/work/cascades/sampanna/ir_backup/mit_etd_data/data/etds'
    convert_pdf_paths_to_images(input_dir, output_dir, temp, cpu_count=cpu_count(), max_pages=1000)
