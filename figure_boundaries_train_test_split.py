import json
import os
import logging
import random

logger = logging.getLogger(__name__)


def split_train_test(figure_boundaries_path: str, train_output_path: str, test_output_path: str,
                     test_split_percent: int = 20):
    figure_boundaries = json.load(open(figure_boundaries_path, mode='r'))
    random.shuffle(figure_boundaries)
    total = len(figure_boundaries)

    if total > 1:
        split_idx = int(total * test_split_percent / 100)
        json.dump(figure_boundaries[:split_idx], open(test_output_path, mode='w'), indent=2)
        json.dump(figure_boundaries[split_idx:], open(train_output_path, mode='w'), indent=2)
    else:
        print("Too less values to split. Exiting.")


if __name__ == "__main__":
    test_split_percent = 20
    random.seed(0)

    IN_DOCKER = os.environ.get('IN_DOCKER', False)

    if IN_DOCKER:
        dataset_dir = '/work/host-input'
    else:
        dataset_dir = '/home/sampanna/workspace/bdts2/deepfigures-results/gold_standard_2'
    figure_boundaries_path = os.path.join(dataset_dir, 'figure_boundaries.json')
    train_path = os.path.join(dataset_dir, 'figure_boundaries_train.json')
    test_path = os.path.join(dataset_dir, 'figure_boundaries_test.json')
    split_train_test(figure_boundaries_path=figure_boundaries_path, train_output_path=train_path,
                     test_output_path=test_path, test_split_percent=test_split_percent)
