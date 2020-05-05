import os
import json
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

date_issued_set = set()
date_accessioned_set = set()
date_available_set = set()
date_copyright_set = set()

handle_dict = json.load(open('data/dates.json'))
for handle in handle_dict:
    handle_obj = handle_dict[handle]
    date_issued_set.add(handle_obj.get('dc.date.issued', ''))
    date_accessioned_set.add(handle_obj.get('dc.date.accessioned', ''))
    date_available_set.add(handle_obj.get('dc.date.available', ''))
    date_copyright_set.add(handle_obj.get('dc.date.copyright', ''))

print(sorted(date_issued_set))  # Ranges from 1876 to 2019
print(sorted(date_accessioned_set))  # Ranges from 2005 to 2020.
print(sorted(date_copyright_set))  # Ranges from 1899 to 2019
# print(sorted(date_available_set))  # Ranges from 2005 to 2020

# Why we chose dc.date.issued?
# Because this date ranges from 876 to 2019. Much more wider than other dates.
