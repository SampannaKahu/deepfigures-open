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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_save_dir',
                        default='/home/sampanna/workspace/bdts2/deepfigures-results/model_checkpoints/377266_arxiv_2020-06-02_22-48-45/',
                        type=str)
    parser.add_argument('--hidden_set_dir',
                        default='/home/sampanna/workspace/bdts2/deepfigures-results/gold_standard_dataset',
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
                                                model_save_dir=args.model_save_dir,
                                                iteration=iteration,
                                                output_json_file_name="figure_boundaries_hidden_set_{}.json".format(
                                                    iteration),
                                                batch_size=args.batch_size)
