import os
import logging
import requests

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


def invoke(url: str) -> requests.Response:
    payload = {}
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:75.0) Gecko/20100101 Firefox/75.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Referer': 'https://dspace.mit.edu/handle/1721.1/7622',
        'Cookie': 'JSESSIONID=EB241F9F97A13066B865B9F83EC33F11',
        'Upgrade-Insecure-Requests': '1',
        'DNT': '1'
    }
    return requests.request("GET", url, headers=headers, data=payload)


def get_file_path(handle: str) -> str:
    return "data/" + handle_to_filename(handle)


def handle_to_filename(handle: str) -> str:
    return handle.replace('/', '_').replace('.', '_') + '.json'


def filename_to_handle(filename: str) -> str:
    return filename.split('.json')[0].replace('_', '.', 1).replace('_', '/', 1)
