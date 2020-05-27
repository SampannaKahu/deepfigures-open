import os
import json
import glob
import logging
import pprint

pp = pprint.PrettyPrinter(indent=2)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

date_fields = {'dc.date.created', 'dc.date', 'dc.date.available', 'dc.date.copyright', 'dc.date.accessioned',
               'dc.date.issued'}

counter = 0

handle_dict = dict()

for metadata_path in list(glob.glob("data/etds/*.json")):
    handle = metadata_path.split('/')[-1].split('.json')[0]
    counter = counter + 1
    metadata = json.load(open(metadata_path))
    first_column = metadata['0']
    for key in first_column:
        if first_column[key] in date_fields:
            handle_obj = handle_dict.get(handle, {})
            # handle_obj.update({first_column[key], metadata['0'][key]})
            handle_obj[first_column[key]] = metadata['1'][key]
            handle_dict[handle] = handle_obj

            # if counter > 23000:
            #     break
            # print(metadata)
            # pp.pprint(metadata)
            # print(metadata_path)
            # break

pp.pprint(handle_dict)

json.dump(handle_dict, open('data/dates.json', mode='w'), indent=2)
