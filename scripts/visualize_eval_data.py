import os
import json
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


def parse_year_tuple(tuple: str) -> list:
    tuple = "(0.4033084398423145, 5, 9, 10, 0.35714285714285715, 0.3333333333333333, 0.3448275862068965)"
    chunks = tuple.strip("(").strip(")").split(",")
    chunks = [chunk.strip() for chunk in chunks]
    return chunks


eval_data = json.load(open('eval_data.json'))
gold_1_eval_data = eval_data['gold_standard_dataset']
# print(gold_1_eval_data)

original_500000 = gold_1_eval_data['original']['500000']
model_377269_arxiv_2020_06_13_02_05_05_503801 = gold_1_eval_data['377269_arxiv_2020-06-13_02-05-05']['503801']

years = set(original_500000.keys()).intersection(model_377269_arxiv_2020_06_13_02_05_05_503801.keys())
win_loss_dict = {'wins': [], 'losses': []}
for year in years:
    original_500000_f1 = float(original_500000[year].split(' ')[-1].split(')')[0])
    # print(original_500000_f1)
    model_377269_arxiv_2020_06_13_02_05_05_503801_f1 = \
        float(model_377269_arxiv_2020_06_13_02_05_05_503801[year].split(' ')[-1].split(')')[0])
    # print(model_377269_arxiv_2020_06_13_02_05_05_503801_f1)
    if model_377269_arxiv_2020_06_13_02_05_05_503801_f1 >= original_500000_f1:
        print("Yay {}".format(year))
        win_loss_dict['wins'].append(year)
    else:
        # print("Nay {}".format(year))
        win_loss_dict['losses'].append(year)

print("Wins: ", sorted(win_loss_dict['wins']))
print("Losses", sorted(win_loss_dict['losses']))

if __name__ == "__main__":
    parsed = parse_year_tuple("")
    print(parsed)
