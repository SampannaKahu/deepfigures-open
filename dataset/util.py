import os
import json
import shutil
import logging
import requests
import tempfile

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


def invoke(url: str) -> requests.Response:
    payload = {}
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:75.0) Gecko/20100101 Firefox/75.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Referer': 'https://dspace.mit.edu/handle/1721.1/7622',
        'Cookie': 'JSESSIONID=EB241F9F97A13066B865B9F83EC33F11',
        'Upgrade-Insecure-Requests': '1',
        'DNT': '1'
    }
    return requests.request("GET", url, headers=headers, data=payload)


def handle_to_pdffilename(handle: str) -> str:
    return handle.replace('/', '_').replace('.', '_') + '.pdf'


def pdffilename_to_handle(filename: str) -> str:
    return filename.split('.pdf')[0].replace('_', '.', 1).replace('_', '/', 1)


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
            # os.rename(image_path, dest_image_path)
            shutil.move(image_path, dest_image_path)
            logger.info("Moved {src} to {dest}".format(src=image_path, dest=dest_image_path))


def filename_to_handle(filename: str) -> str:
    return filename.split('.json')[0].replace('_', '.', 1).replace('_', '/', 1)


def get_handle_from_path(etd_metadata_path: str) -> str:
    filename = etd_metadata_path.split('/')[-1]
    return filename_to_handle(filename)


def read_etd_metadata(metadata_file_path: str) -> dict:
    with open(metadata_file_path) as fp:
        metadata = json.load(fp)
    return metadata


def transform_metadata(metadata: dict) -> dict:
    if '0' not in metadata or '1' not in metadata or '2' not in metadata:
        raise ValueError('This metadata does not seem to be have exactly 3 columns')
    first_col = metadata['0']
    second_col = metadata['1']
    third_col = metadata['2']

    if len(first_col.values()) != len(second_col.values()) or len(first_col.values()) != len(third_col.values()):
        raise ValueError("The column lengths are of un-equal lengths.")

    return dict(zip(first_col.values(), zip(second_col.values(), third_col.values())))


def combine_metadata_files(metadata_file_paths: list, output_path: str):
    all_metadata = {
        get_handle_from_path(metadata_file_path): transform_metadata(read_etd_metadata(metadata_file_path))
        for metadata_file_path in metadata_file_paths
    }
    with open(output_path, mode='w') as all_metadata_fp:
        json.dump(all_metadata, all_metadata_fp, indent=2)
