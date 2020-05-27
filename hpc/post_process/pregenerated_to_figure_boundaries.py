import os
import shutil
import zipfile
import glob
import logging
import torch
import json
import typing
from multiprocessing import Pool
from figure_boundaries_train_test_split import split_train_test

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename=os.path.basename(__file__).split('.')[0] + '.log')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

job_output_directory = '/home/sampanna/deepfigures-results/372902_temp'
dataset_dir = '/home/sampanna/deepfigures-results/arxiv_coco_dataset'
image_save_dir = os.path.join(dataset_dir, 'images')
figure_boundaries_save_path = os.path.join(dataset_dir, 'figure_boundaries.json')
figure_boundaries_train_save_path = os.path.join(dataset_dir, 'figure_boundaries_train.json')
figure_boundaries_test_save_path = os.path.join(dataset_dir, 'figure_boundaries_test.json')
test_split_percent = 20
tmp_extract_dir = os.path.join(dataset_dir, 'tmp')


def unzip_zip_file(zip_file_path: str, extract_dir: str = tmp_extract_dir) -> typing.Tuple[
    typing.List[str], typing.List[str]]:
    """
    Takes in a zip file path. Unzips it in a temporary directory. And returns the list of the files unzipped.
    Works only for flat file-structured zips.
    :param zip_file_path: path os the zip file.
    :param extract_dir: The directory to extract the data into.
    :return: the list of path of the contents of the zip (all, png and pt)
    """
    process_unzip_dir = os.path.join(extract_dir, str(os.getpid()))
    os.makedirs(process_unzip_dir, exist_ok=True)
    zip = zipfile.ZipFile(zip_file_path)
    zip.extractall(path=process_unzip_dir)
    zip.close()
    file_list = os.listdir(os.path.join(process_unzip_dir, 'tmp'))
    png_paths = [os.path.join(process_unzip_dir, 'tmp', path) for path in file_list if '.png' in path]
    pt_paths = [os.path.join(process_unzip_dir, 'tmp', path) for path in file_list if '.pt' in path]
    assert len(png_paths) == len(pt_paths)
    return png_paths, pt_paths


def get_rect(bb_tensor):
    return {
        "x1": bb_tensor[1].item(),
        "x2": bb_tensor[2].item(),
        "y1": bb_tensor[3].item(),
        "y2": bb_tensor[4].item()
    }


os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(image_save_dir, exist_ok=True)
shutil.rmtree(tmp_extract_dir, ignore_errors=True)
os.makedirs(tmp_extract_dir, exist_ok=True)

current_image_id = 1
figure_boundaries = []

zip_paths = glob.glob(os.path.join(job_output_directory, '*/*.zip'), recursive=True)
batch_size = 7
batches = [zip_paths[i:i + batch_size] for i in range(0, len(zip_paths), batch_size)]
for batch in batches:
    p = Pool(batch_size)
    result_list = p.map(unzip_zip_file, batch)
    png_paths = []
    pt_paths = []
    for result_tuple in result_list:
        png_paths = png_paths + result_tuple[0]
        pt_paths = pt_paths + result_tuple[1]

    for idx, png_path in enumerate(png_paths):
        pt_path = pt_paths[idx]
        logger.info("Idx: {}, Png path: {}, pt path: {}.".format(idx, png_path, pt_path))
        if png_path.split('.png')[0] != pt_path.split('.pt')[0]:
            logger.warning("Found an instance when the pt path is not the same as png path. Skipping")
            logger.warning("pt path: {}. Png path: {}".format(pt_path, png_path))
            continue

        _image_name = str(current_image_id) + '.png'
        _image_save_path = os.path.join(image_save_dir, _image_name)
        os.rename(png_path, _image_save_path)
        figure_boundaries.append({
            "image_path": _image_name,
            "rects": [get_rect(bb_tensor) for bb_tensor in torch.load(pt_path)]
        })
        current_image_id = current_image_id + 1

    # Cleanup the temp directory.
    shutil.rmtree(tmp_extract_dir, ignore_errors=True)
    os.makedirs(tmp_extract_dir, exist_ok=True)
    # checkpoint the annotation file.
    json.dump(figure_boundaries, open(figure_boundaries_save_path, mode='w'), indent=2)
    logger.info("Successfully saved annotations after processing zipfile batch paths: {}".format(batch))
    logger.info("Current image id: {}".format(current_image_id))

split_train_test(figure_boundaries_path=figure_boundaries_save_path,
                 train_output_path=figure_boundaries_train_save_path,
                 test_output_path=figure_boundaries_test_save_path, test_split_percent=test_split_percent)
