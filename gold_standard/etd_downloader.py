import logging
import pandas as pd
import os
from lxml import html, etree
import util

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


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
        page = util.invoke("https://dspace.mit.edu/handle/" + self.handle + "?show=full")
        if page.status_code >= 400:
            logger.error("Http call failed. Response code: {}".format(page.status_code))
            return
        tree = html.fromstring(page.content)
        self._download_pdf(tree, self.pdf_save_path)
        self._download_metadata(tree, self.metadata_save_path)

    def _download_pdf(self, tree, save_path: str) -> None:
        download_link_element = tree.xpath("/html/body/div[4]/div[2]/div/div[1]/div/div[1]/div[2]/div/div[3]/a")
        download_url = "https://dspace.mit.edu" + download_link_element[0].attrib['href']
        response = util.invoke(download_url)
        with open(save_path, mode='wb') as save_file:
            save_file.write(response.content)

    def _download_metadata(self, tree, save_path: str) -> None:
        table_xpath = "/html/body/div[4]/div[2]/div/div[1]/div/div[1]/div[1]/table"
        table_element = tree.xpath(table_xpath)
        pretty_table_element = etree.tostring(table_element[0], pretty_print=True)
        table_dfs = pd.read_html(pretty_table_element)
        table_dfs[0].to_json(save_path)


if __name__ == "__main__":
    etd_downloader = EtdDownloader("1721.1/11833", "/tmp", create_dir_if_not_exist=True)
    etd_downloader.download()
