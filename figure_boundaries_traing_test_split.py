import json
import os

test_split_percent = 20

IN_DOCKER = os.environ.get('IN_DOCKER', False)

if IN_DOCKER:
    figure_boundaries_path = '/work/host-input/figure_boundaries.json'
    train_path = '/work/host-input/figure_boundaries_train.json'
    test_path = '/work/host-input/figure_boundaries_test.json'
else:
    figure_boundaries_path = 'figure_boundaries.json'
    train_path = 'figure_boundaries_train.json'
    test_path = 'figure_boundaries_test.json'

figure_boundaries = json.load(open(figure_boundaries_path, mode='r'))

total = len(figure_boundaries)

if total > 1:
    split_idx = int(total * test_split_percent / 100)
    json.dump(figure_boundaries[:split_idx], open(test_path, mode='w'))
    json.dump(figure_boundaries[split_idx:], open(train_path, mode='w'))
else:
    print("Too less values to split. Exiting.")
