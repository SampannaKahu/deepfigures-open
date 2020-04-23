import os
import logging
from lxml import html, etree
import util

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


class BrowseExplorer:
    def __init__(self,
                 handle: str,
                 year: int = 1960,
                 results_per_page: int = 20,
                 sort_by: str = "dateissued",
                 order: str = "ASC") -> None:
        self.handle = handle
        self.results_per_page = results_per_page
        self.sort_by = sort_by
        self.order = order
        self.year = year
        self.browse_url = self._build_url()
        super().__init__()

    def _build_url(self):
        return "https://dspace.mit.edu/handle/" \
               + self.handle \
               + "/browse?rpp=" \
               + str(self.results_per_page) \
               + "&sort_by=2&type=" \
               + self.sort_by \
               + "&etal=-1&order=" \
               + self.order \
               + "&year=" \
               + str(self.year) \
               + "&starts_with=&submit=Go"

    def download_all_handles(self):
        pass

    def download_handles(self):
        page = util.invoke(self.browse_url)
        tree = html.fromstring(page.content)
        element_list = tree.xpath("/html/body/div[4]/div[2]/div/div[1]/div/div/div[3]/ul")
        handles = list()
        for i in range(1, 25):
            li = element_list[0].xpath("li[" + str(i) + "]")
            if not len(li):
                break
            list_ahref = li[0].xpath("div/div[2]/div/h4/a")
            year = int(li[0].xpath("div/div[2]/div/div/span[2]/small/span[2]")[0].text)
            # string = li[0].attrib['href'].split("/handle/")[1]
            handles.append({
                'handle': list_ahref[0].attrib['href'].split("/handle/")[1],
                'year': year
            })

        return handles

    def download_etd(self):
        pass


if __name__ == "__main__":
    browse_explorer = BrowseExplorer(handle="1721.1/7608")
    browse_explorer.download_handles()
