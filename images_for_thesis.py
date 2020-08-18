import os
import glob
import json
import logging
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

random.seed(0)
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


def plot_and_save_image(image_path: str, bbs, save_dir: str) -> None:
    im = np.array(Image.open(image_path), dtype=np.uint8)
    fig, ax = plt.subplots(1)
    ax.imshow(im)
    mng = plt.get_current_fig_manager()
    # mng.frame.Maximize(True)
    mng.resize(*mng.window.maxsize())
    for bb in bbs:
        x1, y1, x2, y2 = bb['x1'], bb['y1'], bb['x2'], bb['y2']
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    save_path = os.path.join(save_dir, os.path.basename(image_path.split('.png')[0] + '_bb.png'))
    plt.savefig(save_path)
    # plt.show()


gold_standard_dataset_dir = '/home/sampanna/workspace/bdts2/deepfigures-results/gold_standard_dataset'
num_images_to_sample = 20

images = glob.glob(os.path.join(gold_standard_dataset_dir, 'images', '*.png'))

annos = json.load(open(os.path.join(gold_standard_dataset_dir, 'figure_boundaries.json')))
print(len(annos))
valid_annos = [anno for anno in annos if anno['rects']]
print(len(valid_annos))
sampled_annos = random.sample(valid_annos, num_images_to_sample)
print(sampled_annos)
print(len(sampled_annos))

for anno in sampled_annos:
    plot_and_save_image(image_path=os.path.join(gold_standard_dataset_dir, 'images', anno['image_path']),
                        bbs=anno['rects'],
                        save_dir='/tmp')

