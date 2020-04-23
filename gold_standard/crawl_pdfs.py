import json
import logging
import csv
import copy
import requests

logging.basicConfig()
logger = logging.getLogger("crawler_log")

#mit_index = json.load(open("mit_depts_with_subcommunities.json"))

# print(mit_index)

# def get_url(handle: str, year: int) -> str:
#     return "https://dspace.mit.edu/handle/" \
#            + sub_comm['handle'] \
#            + "/browse?rpp=100&sort_by=2&type=dateissued&etal=-1&order=ASC&year=" \
#            + str(year) \
#            + "&starts_with=&submit=Go"
#
#
# csv_row = {
#     'Department name': '',
#     'Degree type': '',
#     'Handle': '',
#     'Browse URL': '',
#     'Year': ''
# }
#
# with open("data_labelling.csv", mode='w') as csvfile:
#     # field_names = ['Department name', 'Degree type', 'Handle', 'Browse Url', 'Year']
#     field_names = csv_row.keys()
#     writer = csv.DictWriter(csvfile, fieldnames=field_names)
#     writer.writeheader()
#     for dept in mit_index:
#         degree_types = dict()
#         for sub_comm in dept['sub_communities']:
#             year = 1960
#             handle = sub_comm['handle']
#             url = get_url(handle, year)
#             print(sub_comm['name'].split(" - ")[0], sub_comm['degree_type'], sub_comm['handle'], url)
#             writer.writerow({
#                 'Department name': sub_comm['name'].split(" - ")[0],
#                 'Degree type': sub_comm['degree_type'],
#                 'Handle': sub_comm['handle'],
#                 'Browse URL': url,
#                 'Year': ''
#             })


if __name__ == "__main__":
    pass
