import os
import json
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

handle_dict = json.load(open('data/dates.json'))
date_issued_to_handle_dict = dict()
for handle in handle_dict:
    handle_obj = handle_dict[handle]
    if 'dc.date.issued' in handle_obj:
        date_issued = handle_obj.get('dc.date.issued')
        date_issued_handles = date_issued_to_handle_dict.get(date_issued, [])
        date_issued_handles.append(handle)
        date_issued_to_handle_dict[date_issued] = date_issued_handles

print(date_issued_to_handle_dict)

date_issued_histogram = dict()
for date_issued in date_issued_to_handle_dict:
    # print(date_issued, len(date_issued_to_handle_dict[date_issued]))
    date_issued_histogram[date_issued] = len(date_issued_to_handle_dict[date_issued])
