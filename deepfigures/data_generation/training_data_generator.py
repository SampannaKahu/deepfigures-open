import os
import cv2
import json
import argparse
import logging
import torch
from deepfigures.data_generation.arxiv_dataset import ArxivDataSet
from deepfigures import settings
from torch.utils.data.dataloader import DataLoader
from zipfile import ZipFile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_list_json', type=str, default='/home/sampanna/deepfigures-results/files.json',
                        help='Provide the path to a json file which contains the list of all files to be procesed.')
    parser.add_argument('--n_cpu', type=int, default=settings.PROCESS_PAPER_TAR_THREAD_COUNT,
                        help='number of workers')
    parser.add_argument('--images_per_zip', type=int, default=500, help='Maximum number of images per zip file.')
    parser.add_argument('--zip_save_dir', type=str, default='/tmp',
                        help='Provide the path where all the generated zips will be saved.')
    parser.add_argument('--work_dir_prefix', type=str, default=settings.HOSTNAME,
                        help='The prefix for the work directory of each instance of ArxivDataset.')
    parser.add_argument('--arxiv_tmp_dir', type=str, default=settings.ARXIV_DATA_TMP_DIR,
                        help='The tmp directory for arxiv data.')
    parser.add_argument('--arxiv_cache_dir', type=str, default=settings.ARXIV_DATA_CACHE_DIR,
                        help='The download cache for arxiv data.')
    parser.add_argument('--arxiv_data_output_dir', type=str, default=settings.ARXIV_DATA_OUTPUT_DIR,
                        help='The output directory for arxiv data.')
    parser.add_argument('--delete_tar_after_extracting', type=bool, default=False,
                        help='Whether to delete the original tar file from the download cache or not. ' +
                             'e.g. s3://arxiv/src/arXiv_src_0003_001.tar')
    parser.add_argument('--augment_typewriter_font', type=bool, default=False,
                        help='Whether to augment the data with typewriter font.')
    parser.add_argument('--augment_line_spacing_1_5', type=bool, default=False,
                        help='Whether to augment the data with 1.5 line spacing.')
    parser.add_argument('--should_apply_image_based_augmentations', type=bool, default=False,
                        help='If set to true, applies all the default image-based augmentations.')

    return parser.parse_args()


def save_tensor_image(my_tensor: torch.tensor, save_path: str):
    permuted_image = my_tensor.permute(1, 2, 0)
    cv2.imwrite(save_path, permuted_image.numpy())


def plot_png_image(path: str):
    img = mpimg.imread(path)
    plt.imshow(img)
    plt.show()


def get_zipfile_name(zip_file_id: int) -> str:
    return settings.HOSTNAME + '_' + str(zip_file_id) + '.zip'


if __name__ == "__main__":
    """
    Command to invoke:
    /home/sampanna/.conda/envs/deepfigures/bin/python /home/sampanna/deepfigures-open/deepfigures/data_generation/training_data_generator.py \
        --file_list_json /home/sampanna/deepfigures-open/hpc/files_random_40/files_"$i".json \
        --images_per_zip=500 \
        --zip_save_dir=/work/cascades/sampanna/deepfigures-results/pregenerated_training_data/"$SYSNAME"_"$SLURM_JOBID"_"$i"_"$ts" \
        --n_cpu=1 \
        --work_dir_prefix "$SYSNAME"_"$SLURM_JOBID"_"$i"_"$ts" \
        --arxiv_tmp_dir /scratch-ssd/arxiv_data_temp \
        --arxiv_cache_dir /scratch-ssd/download_dache \
        --arxiv_data_output_dir /scratch-ssd/arxiv_data_output \
        --delete_tar_after_extracting True \
        --augment_typewriter_font True \
        --augment_line_spacing_1_5 True
    """
    args = parse_args()
    logger.info("Parsed arguments: " + str(args))
    print("Parsed arguments: ", args)
    os.makedirs(args.zip_save_dir, exist_ok=True)
    input_files = json.load(open(args.file_list_json))
    if args.should_apply_image_based_augmentations:
        image_based_transformations = settings.seq
    else:
        image_based_transformations = settings.no_op
    dataset = ArxivDataSet(list_of_files=input_files, shuffle_input=True, work_dir_prefix=args.work_dir_prefix,
                           arxiv_tmp_dir=args.arxiv_tmp_dir, arxiv_cache_dir=args.arxiv_cache_dir,
                           arxiv_data_output_dir=args.arxiv_data_output_dir, get_raw_image=True,
                           delete_tar_after_extracting=args.delete_tar_after_extracting,
                           augment_typewriter_font=args.augment_typewriter_font,
                           augment_line_spacing_1_5=args.augment_line_spacing_1_5,
                           image_augmentation_transform_sequence=settings.no_op)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.n_cpu)
    data_iterator = iter(data_loader)

    zip_file_id = 0
    z = ZipFile(os.path.join(args.zip_save_dir, get_zipfile_name(zip_file_id)), mode='w')

    file_counter = 0
    for batch in data_iterator:
        imgs, targets, _, _ = batch

        # save to temp files.
        image_png_path = '/tmp/' + str(file_counter) + '.png'
        target_path = '/tmp/' + str(file_counter) + '.pt'
        save_tensor_image(imgs[0], image_png_path)
        torch.save(targets[0], target_path)

        # add temp files to the zip file.
        z.write(image_png_path)
        z.write(target_path)

        # delete the temporary files.
        os.remove(image_png_path)
        os.remove(target_path)

        # increment counter.
        file_counter = file_counter + 1

        # if zip file reached threshold, rollover to the next one.
        if file_counter == args.images_per_zip:
            z.close()
            print("Finished zip file number {}.".format(zip_file_id))
            file_counter = 0
            zip_file_id = zip_file_id + 1
            z = ZipFile(os.path.join(args.zip_save_dir, get_zipfile_name(zip_file_id)), mode='w')
    z.close()
