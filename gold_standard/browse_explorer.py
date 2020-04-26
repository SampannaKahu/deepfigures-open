import os
import logging
from lxml import html
import util
import json
from typing import List
import time

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


def get_sub_community_handles(mit_index_json_file: str) -> List[str]:
    sub_community_handles = []
    with open(mit_index_json_file) as fp:
        mit_index = json.load(fp)
    for dept in mit_index:
        for sub_community in dept['sub_communities']:
            sub_community_handles.append(sub_community['handle'])
    return sub_community_handles


def get_total_items(collection_handle: str) -> int:
    """
    Sample URL:
    https://dspace.mit.edu/handle/1721.1/7608/browse?rpp=20&offset=1&etal=-1&sort_by=2&type=dateissued&order=ASC
    :param collection_handle:
    :return: an integer with the total number of items in this collection.
    """
    url = "https://dspace.mit.edu/handle/" \
          + collection_handle \
          + "/browse?rpp=20&offset=1&etal=-1&sort_by=2&type=dateissued&order=ASC"
    page = util.invoke(url)
    if page.status_code >= 400:
        logger.error("Received status code {} for url {}".format(page.status_code, url))
        return 0
    tree = html.fromstring(page.content)
    pagination_element = tree.xpath("/html/body/div[4]/div[2]/div/div[1]/div/div/div[2]/div/div[1]/p")
    if not len(pagination_element):
        return 0
    return int(pagination_element[0].text.split(" ")[-1])


def get_handles_on_browse_page(collection_handle: str, offset: int, rpp: int = 20) -> List[dict]:
    browse_url = "https://dspace.mit.edu/handle/" + collection_handle + "/browse?rpp=" + str(rpp) + "&offset=" \
                 + str(offset) + "&etal=-1&sort_by=2&type=dateissued&order=ASC"
    page = util.invoke(browse_url)
    if page.status_code >= 400:
        logger.error("Received status code {} for url {}".format(page.status_code, browse_url))
        return []
    element_list = html.fromstring(page.content) \
        .xpath("/html/body/div[4]/div[2]/div/div[1]/div/div/div[3]/ul")
    handles = list()
    counter = 1
    while True:
        li = element_list[0].xpath("li[" + str(counter) + "]")
        if not len(li):
            break
        try:
            handles.append({
                'handle': li[0].xpath("div/div[2]/div/h4/a")[0].attrib['href'].split("/handle/")[1],
                'year': int(li[0].xpath("div/div[2]/div/div/span[2]/small/span[2]")[0].text)
            })
        except ValueError:
            handles.append({
                'handle': li[0].xpath("div/div[2]/div/h4/a")[0].attrib['href'].split("/handle/")[1],
                'year': li[0].xpath("div/div[2]/div/div/span[2]/small/span[2]")[0].text
            })
        counter = counter + 1

    return handles


def download_handles_in_collection(collection_handle: str, stop_year: int = 1999, rpp: int = 100) -> List[str]:
    print("Downloading handles for collection {}".format(collection_handle))
    offset = 1
    handles = []
    total_items = get_total_items(collection_handle)
    while len(handles) < total_items and (len(handles) == 0 or handles[-1]['year'] < stop_year):
        time.sleep(5)
        new_handles = get_handles_on_browse_page(collection_handle, offset, rpp)
        if not len(new_handles):
            break
        handles = handles + new_handles
        offset = offset + rpp
        print("Total handles downloaded: ", len(handles))
    print("Downloaded {} handles for collection {}.".format(str(len(handles)), collection_handle))
    return handles


if __name__ == "__main__":
    for handle in get_sub_community_handles("mit_depts_with_subcommunities.json"):
        time.sleep(5)
        if os.path.exists(util.get_file_path(handle)):
            print("Data file for {} exists. Skipping.".format(handle))
            continue
        handles = download_handles_in_collection(handle)
        if handles:
            with open(util.get_file_path(handle), mode='w') as fp:
                json.dump(handles, fp)
        else:
            logger.error("Got empty list for {}. Skipping saving.".format(handle))
