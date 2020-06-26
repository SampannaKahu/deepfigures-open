import os
import logging
import argparse
from deepfigures.extraction import detection
from tqdm import tqdm
from pprint import pformat

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    """
    Command:
    python -c 'from deepfigures.extraction import detection; detection.run_detection_on_coco_dataset("/home/sampanna/deepfigures-results/gold_standard_dataset", "images", "figure_boundaries.json", "/home/sampanna/deepfigures-results/model_checkpoints/377269_arxiv_2020-06-13_02-05-05/", 550000, "figure_boundaries_hidden_set_550000.json", 1000)'
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_save_dir',
                        default='/home/sampanna/deepfigures-results/model_checkpoints/377268_arxiv_2020-06-14_01-23-25/',
                        type=str)
    parser.add_argument('--hidden_set_dir',
                        default='/home/sampanna/deepfigures-results/gold_standard_dataset',
                        type=str)
    parser.add_argument('--figure_boundaries_file_name',
                        default='figure_boundaries.json',
                        type=str)
    parser.add_argument('--batch_size',
                        default=1000,
                        type=int)
    args = parser.parse_args()
    ckpt_iter_numbers = [int(ckpt.strip().split(' ')[1].strip('"').split('-')[-1]) for ckpt in
                         open(os.path.join(args.model_save_dir, 'checkpoint')).readlines()][1:]
    print("Args: {}".format(pformat(args)))
    print("Checkpoint iters: {}".format(ckpt_iter_numbers))
    for iteration in tqdm(ckpt_iter_numbers):
        detection.run_detection_on_coco_dataset(dataset_dir=args.hidden_set_dir,
                                                images_sub_dir='images',
                                                figure_boundaries_file_name=args.figure_boundaries_file_name,
                                                model_save_dir=args.model_save_dir,
                                                iteration=iteration,
                                                output_json_file_name="figure_boundaries_hidden_set_{}.json".format(
                                                    iteration),
                                                batch_size=args.batch_size)
