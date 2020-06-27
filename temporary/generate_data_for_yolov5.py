import os
import glob
import shutil
import argparse
import random
import tempfile
import zipfile
import torch
from PIL import Image
import multiprocessing
from functools import partial


# logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(os.path.basename(__file__))
# logger.setLevel(logging.DEBUG)

# zip_dir = '/work/cascades/sampanna/deepfigures-results/pregenerated_training_data/377269'
# zip_dir = '/home/sampanna/Desktop/data'
# tmp_dir = '/tmp'
# output_dir = '/tmp/data'
# num_zips_to_process = 2
# random_seed = 42


def process_zip(zip_path: str, tmp_dir: str, output_dir: str) -> None:
    print(zip_path)
    output_file_prefix = os.path.basename(zip_path).split('.zip')[0]
    with tempfile.TemporaryDirectory(dir=tmp_dir, prefix=output_file_prefix) as td:
        zip = zipfile.ZipFile(zip_path)
        zip.extractall(path=td)
        zip.close()
        file_list = os.listdir(os.path.join(td, 'tmp'))
        png_paths = [os.path.join(td, 'tmp', path) for path in file_list if '.png' in path]
        pt_paths = [os.path.join(td, 'tmp', path) for path in file_list if '.pt' in path]
        png_paths = sorted(png_paths)
        pt_paths = sorted(pt_paths)
        assert len(png_paths) == len(pt_paths)
        for idx, png_path in enumerate(png_paths):
            pt_path = pt_paths[idx]
            if png_path.split('.png')[0] != pt_path.split('.pt')[0]:
                print("Found an instance when the pt path is not the same as png path. Skipping")
                print("pt path: {}. Png path: {}".format(pt_path, png_path))
                continue

            # get the width and height of image.
            im = Image.open(png_path)
            width, height = im.size

            # move the image file
            dest_image_path = os.path.join(output_dir, output_file_prefix + '_' + str(idx) + '.png')
            shutil.move(png_path, dest_image_path)

            rects = [
                ' '.join([
                    '0',  # class id
                    str((bb_tensor[2].item() + bb_tensor[1].item()) / (2 * width)),  # xc
                    str((bb_tensor[4].item() + bb_tensor[3].item()) / (2 * height)),  # yc
                    str((bb_tensor[2].item() - bb_tensor[1].item()) / width),  # bb_width
                    str((bb_tensor[4].item() - bb_tensor[3].item()) / height)  # bb_height
                ])
                for bb_tensor in torch.load(pt_path)]
            if rects:
                anno_path = os.path.join(output_dir, output_file_prefix + '_' + str(idx) + '.txt')
                with open(anno_path, mode='w') as anno_file:
                    anno_file.write('\n'.join(rects) + '\n')


if __name__ == "__main__":
    """
    python generate_data_for_yolov5.py \
    --zip_dir='/work/cascades/sampanna/deepfigures-results/pregenerated_training_data/377269' \
    --tmp_dir='/tmp' \
    --output_dir='/work/cascades/sampanna/yolov5/data/377269' \
    --num_zips_to_process=1000 \
    --random_seed=42 \
    --n_cpu=5
    
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--zip_dir', required=True, type=str)
    parser.add_argument('--tmp_dir', default="/tmp", type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--num_zips_to_process', type=int, default=1000)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--n_cpu', type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()

    os.makedirs(args.output_dir)
    random.seed(args.random_seed)
    zip_paths = glob.glob(os.path.join(args.zip_dir, '*.zip'))
    random.shuffle(zip_paths)

    zip_paths = zip_paths[:args.num_zips_to_process]
    print(zip_paths)

    with multiprocessing.Pool(args.n_cpu) as pool:
        pool.map(partial(process_zip, tmp_dir=args.tmp_dir, output_dir=args.output_dir), zip_paths)
