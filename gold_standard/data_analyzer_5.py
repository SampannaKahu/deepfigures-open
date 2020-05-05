import os
import json
import random
import logging
import util

random.seed(42)
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

with open('data/all_metadata.json') as fp:
    all_metadata = json.load(fp)


def get_year_issued(all_metadata: dict, handle: str) -> int:
    # try:
    issued_str = all_metadata[handle]['dc.date.issued'][0]
    return int(issued_str.split('-')[0])
    # return int(issued_str)
    # except TypeError:
    #     return 0


with open('mit_depts_with_subcommunities.json') as fp:
    mit_index = json.load(fp)

with open('data/sampled_etds.json') as fp:
    sampled_etds = json.load(fp)

issue_date_upper = 1990  # inclusive
issue_date_lower = 1890  # inclusive

for dept in mit_index:
    for sub_comm in dept['sub_communities']:
        sub_comm_handle = sub_comm['handle']
        # print(sub_comm_handle)
        sub_comm_json_path = 'data/' + util.handle_to_filename(sub_comm_handle)
        if not os.path.isfile(sub_comm_json_path):
            logger.warning("Skipping sub comm: {} since no file found".format(sub_comm_handle))
            continue
        with open(sub_comm_json_path) as fp:
            sub_comm_etd_handles = [etd_obj for etd_obj in json.load(fp) if get_year_issued(all_metadata, etd_obj[
                'handle']) <= issue_date_upper and get_year_issued(all_metadata, etd_obj['handle']) >= issue_date_lower]
            if not sub_comm_etd_handles:
                logger.warning("Skipping sub comm: {} since no applicable etd found".format(sub_comm_handle))
                continue
            chosen_etd = random.choice(sub_comm_etd_handles)
            sampled_etds.append(chosen_etd['handle'])

print(len(sampled_etds))
print(sampled_etds)

json.dump(sampled_etds, open('data/sampled_etds.json', mode='w'))
