import logging
import pandas as pd
import os
from lxml import html, etree
import util
import json
from typing import List
from time import sleep

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_name = str(os.path.basename(__file__).split(".")[0])
logger = logging.getLogger(file_name)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler(file_name + '.log'))


class EtdDownloader(object):
    """
    Given the handle for an ETD, downloads an ETD (pdf and metadata).
    """

    def __init__(self, handle: str, save_dir: str, create_dir_if_not_exist: bool = True) -> None:
        super().__init__()
        if not handle:
            raise ValueError("Handle should be valid.")
        self.handle = handle
        self.save_dir = save_dir
        self.create_dir_if_not_exist = create_dir_if_not_exist
        if not os.path.exists(save_dir) or not os.path.isdir(save_dir):
            if self.create_dir_if_not_exist:
                os.makedirs(save_dir, exist_ok=True)
            else:
                raise ValueError("save_dir does not exist or is not a directory.")
        self.pdf_save_path = os.path.join(save_dir, handle.replace("/", "_") + ".pdf")
        self.metadata_save_path = os.path.join(save_dir, handle.replace("/", "_") + ".xml")

    def download(self) -> None:
        if os.path.exists(self.pdf_save_path) or os.path.exists(self.metadata_save_path):
            logger.info("Skipping already downloaded etd {}.".format(self.handle))
            return
        page = util.invoke("https://dspace.mit.edu/handle/" + self.handle + "?show=full")
        while page.status_code >= 400:
            logger.error("Http call failed. Code: {}. Sleeping for {} secs.".format(page.status_code, 5))
            sleep(5)
            page = util.invoke("https://dspace.mit.edu/handle/" + self.handle + "?show=full")
        tree = html.fromstring(page.content)
        self._download_pdf(tree, self.pdf_save_path)
        self._download_metadata(tree, self.metadata_save_path)

    def _download_pdf(self, tree, save_path: str) -> None:
        download_link_element = tree.xpath("/html/body/div[4]/div[2]/div/div[1]/div/div[1]/div[2]/div/div[3]/a")
        download_url = "https://dspace.mit.edu" + download_link_element[0].attrib['href']
        response = util.invoke(download_url)
        while response.status_code >= 400:
            logger.error("Http call failed. Code: {}. Sleeping for {} secs.".format(response.status_code, 5))
            sleep(5)
            response = util.invoke("https://dspace.mit.edu/handle/" + self.handle + "?show=full")
        with open(save_path, mode='wb') as save_file:
            save_file.write(response.content)

    def _download_metadata(self, tree, save_path: str) -> None:
        table_xpath = "/html/body/div[4]/div[2]/div/div[1]/div/div[1]/div[1]/table"
        table_element = tree.xpath(table_xpath)
        pretty_table_element = etree.tostring(table_element[0], pretty_print=True)
        table_dfs = pd.read_html(pretty_table_element)
        table_dfs[0].to_json(save_path)


def get_sub_communities() -> List[str]:
    with open("mit_depts_with_subcommunities.json") as fp:
        mit_index = json.load(fp)

    handles = []
    for dept in mit_index:
        for sub_com in dept['sub_communities']:
            handles.append(sub_com['handle'])
    return handles


def get_etd_in_sub_community(sub_comm_handle: str) -> List[dict]:
    with open(util.get_file_path(sub_comm_handle)) as fp:
        return json.load(fp)


if __name__ == "__main__":
    for sub_comm_handle in get_sub_communities():
        logger.info("Starting the sub community: {}".format(sub_comm_handle))
        if not os.path.exists(util.get_file_path(sub_comm_handle)):
            logger.info("No file found for sub community {}. Skipping.".format(sub_comm_handle))
            continue
        etds_in_sub_community = get_etd_in_sub_community(sub_comm_handle)
        if not etds_in_sub_community:
            logger.info("No etd in sub community {}. Skipping.".format(sub_comm_handle))
            continue
        for etd_handle_obj in etds_in_sub_community:
            logger.info("Starting etd: {}".format(etd_handle_obj['handle']))
            etd_downloader = EtdDownloader(etd_handle_obj['handle'], "data/etds", create_dir_if_not_exist=True)
            etd_downloader.download()
