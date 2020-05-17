import os
import re
import io
import zipfile
import glob
import logging
import torch
from PIL import Image
from draw_boxes_on_results import plot_bbs

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

job_output_directory = '/home/sampanna/deepfigures-results/pregenerated_training_data'
image_save_dir = '/home/sampanna/arxiv_coco_dataset'


def atoi(text):
    """
    Credits: https://stackoverflow.com/a/18415320/2554177
    :param text:
    :return:
    """
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    Credits: https://stackoverflow.com/a/18415320/2554177
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c.lower()) for c in re.split(r'(\d+)', text)]


def get_next_image_id(image_dir: str) -> int:
    if not os.path.exists(image_dir):
        return 0

    if len(os.listdir(image_dir)) == 0:
        return 0

    image_paths = glob.glob(os.path.join(image_dir, '**'))
    return int(max(image_paths, key=natural_keys).split('/')[-1].split('.')[0]) + 1


id = get_next_image_id(image_save_dir)

for path in glob.glob(os.path.join(job_output_directory, '*/*.zip'), recursive=True):
    zip = zipfile.ZipFile(path)
    png_paths = [path for path in zip.namelist() if '.png' in path]
    pt_paths = [path for path in zip.namelist() if '.pt' in path]
    assert len(png_paths) == len(pt_paths)

    png_paths.sort()
    pt_paths.sort()

    for idx, png_path in enumerate(png_paths):
        pt_path = pt_paths[idx]
        print(idx, png_path, pt_path)
        assert png_path.split('.png')[0] == pt_path.split('.pt')[0]
        with zip.open(png_path) as fp:
            img = Image.open(fp)
            img.save(os.path.join(image_save_dir, str(id) + '.png'))
        with zip.open(pt_path) as fp:
            tnsr = torch.load(io.BytesIO(fp.read()))
        assert tnsr.size() == (1, 5)
        # bb = {
        #     'x1': tnsr[1],
        #     'y1': tnsr[2],
        #     'x2': tnsr[3],
        #     'y2': tnsr[4],
        #
        # }
        # plot_bbs(os.path.join(image_save_dir, '0.png'), [bb])
        id = id + 1

    break
