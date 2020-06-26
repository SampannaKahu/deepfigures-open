import os
import re
import csv
import logging
import argparse
from pprint import pformat

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_log_file', type=str, required=True, help='Provide the path to training log file.')
    parser.add_argument('--val_csv_path', type=str, required=True, help='Provide the path to output val csv file.')
    parser.add_argument('--test_csv_path', type=str, required=True, help='Provide the path to output test csv file.')
    parser.add_argument('--year', type=int, default=0, help='Year for which to generate the csv.')
    return parser.parse_args()


def parse_log_file_and_save_csv(train_log_file: str, val_csv_path: str, test_csv_path: str, year: int):
    with open(train_log_file) as train_log_fp:
        train_logs_lines = train_log_fp.readlines()

    with open(val_csv_path, mode='w') as val_fp:
        with open(test_csv_path, mode='w') as test_fp:
            fieldnames = ["year",
                          "training_step",
                          "mean_iou",
                          "true_positives",
                          "false_positives",
                          "false_negatives",
                          "precision",
                          "recall",
                          "f1",
                          "lr"]
            val_writer = csv.DictWriter(val_fp, fieldnames=fieldnames)
            val_writer.writeheader()
            test_writer = csv.DictWriter(test_fp, fieldnames=fieldnames)
            test_writer.writeheader()
            val = True
            lr = 0
            for log_line in train_logs_lines:
                if 'Evaluating val detection results' in log_line:
                    val = True
                if 'Evaluating test detection results' in log_line:
                    val = False
                if 'Local step' in log_line:
                    lr = float(log_line.split(' lr: ')[1].split(',')[0])
                    global_step = float(log_line.split(' Global step: ')[1].split(',')[0])

                regex_result = re.findall(
                    r"\d{0,4} \(\d?.?\d+, \d?.?\d+, \d?.?\d+, \d?.?\d+, \d?.?\d+, \d?.?\d+, \d?.?\d+\)", log_line)
                if regex_result:
                    num_list = regex_result[0].replace(' ', '').replace('(', ',').replace(')', '').split(',')
                    row = {
                        "year": int(num_list[0]),
                        "training_step": int(global_step),
                        "mean_iou": float(num_list[1]),
                        "true_positives": int(num_list[2]),
                        "false_positives": int(num_list[3]),
                        "false_negatives": int(num_list[4]),
                        "precision": float(num_list[5]),
                        "recall": float(num_list[6]),
                        "f1": float(num_list[7]),
                        "lr": lr
                    }
                    if row["year"] == year:
                        if val:
                            val_writer.writerow(rowdict=row)
                        else:
                            test_writer.writerow(rowdict=row)


if __name__ == '__main__':
    """
    Usage:
    python train_log_parse.py --train_log_file=/home/sampanna/Desktop/train.log --val_csv_path=val.csv --test_csv_path=test.csv
    """
    args = parse_args()
    print("Args: {args}".format(args=pformat(args, indent=2)))
    parse_log_file_and_save_csv(args.train_log_file, args.val_csv_path, args.test_csv_path, args.year)
