import os
import json
import glob
import util
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


def get_handle_from_path(etd_metadata_path: str) -> str:
    filename = etd_metadata_path.split('/')[-1]
    return util.filename_to_handle(filename)


def read_etd_metadata(metadata_file_path: str) -> dict:
    with open(metadata_files_path) as fp:
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


all_metadata = dict()

metadata_files_paths = glob.glob('data/etds/*.json')
for metadata_files_path in metadata_files_paths:
    etd_handle = get_handle_from_path(metadata_files_path)
    metadata = read_etd_metadata(metadata_files_path)
    transformed_metadata = transform_metadata(metadata)
    all_metadata[etd_handle] = transformed_metadata

with open('data/all_metadata.json', mode='w') as all_metadata_fp:
    json.dump(all_metadata, all_metadata_fp, indent=2)
