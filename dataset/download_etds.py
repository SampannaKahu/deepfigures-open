import os
import glob
import logging
import pandas as pd
from tqdm import tqdm
from time import sleep
from lxml import html, etree
from dataset.renderers import GhostScriptRenderer
from dataset.util import invoke, pdffilename_to_handle, convert_pdf_to_images, combine_metadata_files

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


class EtdDownloader(object):
    """
    Given the handle for an ETD, downloads an ETD (pdf and metadata).
    """

    def __init__(self, _handle: str, save_dir: str, create_dir_if_not_exist: bool = True) -> None:
        super().__init__()
        if not _handle:
            raise ValueError("Handle should be valid.")
        self.handle = _handle
        self.save_dir = save_dir
        self.create_dir_if_not_exist = create_dir_if_not_exist
        if not os.path.exists(save_dir) or not os.path.isdir(save_dir):
            if self.create_dir_if_not_exist:
                os.makedirs(save_dir, exist_ok=True)
            else:
                raise ValueError("save_dir does not exist or is not a directory.")
        self.pdf_save_path = os.path.join(save_dir, _handle.replace("/", "_").replace(".", "_") + ".pdf")
        self.metadata_save_path = os.path.join(save_dir, _handle.replace("/", "_").replace(".", "_") + ".json")

    def download(self) -> None:
        if os.path.exists(self.pdf_save_path) or os.path.exists(self.metadata_save_path):
            logger.info("Skipping already downloaded etd {}.".format(self.handle))
            return
        page = invoke("https://dspace.mit.edu/handle/" + self.handle + "?show=full")
        while page.status_code >= 400:
            logger.error("Http call failed. Code: {}. Sleeping for {} secs.".format(page.status_code, 5))
            sleep(5)
            page = invoke("https://dspace.mit.edu/handle/" + self.handle + "?show=full")
        tree = html.fromstring(page.content)
        self._download_pdf(tree, self.pdf_save_path)
        self._download_metadata(tree, self.metadata_save_path)

    def _download_pdf(self, tree, save_path: str) -> None:
        download_link_element = tree.xpath("/html/body/div[4]/div[2]/div/div[1]/div/div[1]/div[2]/div/div[3]/a")
        download_url = "https://dspace.mit.edu" + download_link_element[0].attrib['href']
        response = invoke(download_url)
        while response.status_code >= 400:
            logger.error("Http call failed. Code: {}. Sleeping for {} secs.".format(response.status_code, 5))
            sleep(5)
            response = invoke("https://dspace.mit.edu/handle/" + self.handle + "?show=full")
        with open(save_path, mode='wb') as save_file:
            save_file.write(response.content)

    def _download_metadata(self, tree, save_path: str) -> None:
        table_xpath = "/html/body/div[4]/div[2]/div/div[1]/div/div[1]/div[1]/table"
        table_element = tree.xpath(table_xpath)
        pretty_table_element = etree.tostring(table_element[0], pretty_print=True)
        table_dfs = pd.read_html(pretty_table_element)
        table_dfs[0].to_json(save_path)


if __name__ == "__main__":
    # Read ETD handles.
    with open('etd_urls.txt') as etd_urls_fp:
        handles = [line.strip().split('handle/')[1] for line in etd_urls_fp.readlines() if line]

    # Download ETDs (PDF files and JSON metadata files.)
    for handle in tqdm(handles, desc="Downloading ETDs"):
        EtdDownloader(handle, "etds", create_dir_if_not_exist=True).download()

    # Verify if all ETDs were downloaded.
    pdf_paths = glob.glob('etds/*.pdf')
    downloaded_handles = sorted([pdffilename_to_handle(os.path.basename(pdf_path)) for pdf_path in pdf_paths])
    if sorted(handles) == downloaded_handles:
        print("Successfully downloaded all ETDs.")

    # Merge JSON metadata files into a single file.
    metadata_files_paths = glob.glob('etds/*.json')
    combine_metadata_files(metadata_files_paths, 'metadata.json')

    # Convert PDF files into images of pages.
    _pdf_renderer = GhostScriptRenderer()
    for pdf_path in pdf_paths:
        convert_pdf_to_images(pdf_path, "images", _pdf_renderer, _temp_dir="/tmp", max_pages=500)
