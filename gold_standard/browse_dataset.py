import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.axes import *

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


class CocoDatasetBrowser(object):

    def __init__(self, dataset_dir: str, images_sub_dir: str = 'images', start_index: int = 0) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.images_sub_dir = images_sub_dir
        self.start_index = start_index
        self.figure_boundaries = json.load(open(os.path.join(self.dataset_dir, "figure_boundaries.json")))
        self.current_index = max(min(self.start_index, len(self.figure_boundaries) - 1), 0)
        self.fig, self.ax = plt.subplots(1)
        self.fig.canvas.mpl_connect('key_release_event', self.on_button_release)
        self.current_figure = self.figure_boundaries[self.current_index]
        self.plot_current_fig()

    def set_current_index(self, new_index: int) -> None:
        self.current_index = max(min(new_index, len(self.figure_boundaries) - 1), 0)
        self.current_figure = self.figure_boundaries[self.current_index]

    def plot_current_fig(self) -> None:
        plt.cla()
        im = np.array(
            Image.open(os.path.join(self.dataset_dir, self.images_sub_dir, self.current_figure['image_path'])),
            dtype=np.uint8
        )
        self.ax.imshow(im)
        for bb in self.current_figure['rects']:
            x1, y1, x2, y2 = bb['x1'], bb['y1'], bb['x2'], bb['y2']
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='g', facecolor='none')
            self.ax.add_patch(rect)
        plt.title(self.current_figure['image_path'])
        plt.show()

    def on_button_release(self, event):
        if event.key == 'right':
            self.set_current_index(self.current_index + 1)
            self.plot_current_fig()
        elif event.key == 'left':
            self.set_current_index(self.current_index - 1)
            self.plot_current_fig()


if __name__ == "__main__":
    coco_dataset_browser = CocoDatasetBrowser(
        dataset_dir="/home/sampanna/workspace/bdts2/deepfigures-results/arxiv_coco_dataset"
    )
