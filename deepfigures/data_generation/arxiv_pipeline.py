import os
import glob
import datetime
import tempfile
import tarfile
import logging
import multiprocessing
import multiprocessing.pool
from multiprocessing.dummy import Pool as ThreadPool
import re
import time
import functools
import collections
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import imread
from scipy.misc import imsave
from skimage import measure
from PIL import Image
import scipy as sp
import bs4

import imageio
import imgaug as ia
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from deepfigures import settings
from deepfigures.utils import file_util, config, settings_utils
from deepfigures.extraction import figure_utils, renderers
from deepfigures.extraction.figure_utils import Figure, BoxClass

pdf_renderer = settings_utils.import_setting(
    settings.DEEPFIGURES_PDF_RENDERER)()

IMPORT_STR = r'''
\usepackage{color}
\usepackage{floatrow}
\usepackage{tcolorbox}

\DeclareColorBox{figurecolorbox}{\fcolorbox{%s}{white}}
\DeclareColorBox{tablecolorbox}{\fcolorbox{%s}{white}}

\floatsetup[figure]{framestyle=colorbox, colorframeset=figurecolorbox, framearound=all, frameset={\fboxrule1pt\fboxsep0pt}}
\floatsetup[table]{framestyle=colorbox, colorframeset=tablecolorbox, framearound=all, frameset={\fboxrule1pt\fboxsep0pt}}

\usepackage[labelfont={color=%s},textfont={color=%s}]{caption}

\renewcommand\ttdefault{cmvtt}
\renewcommand{\familydefault}{\ttdefault}
\linespread{1.5}

'''

DOC_CLASS_REGEX = r'\\documentclass\[(.*?)\]\{(.*?)\}'

BEGIN_DOC = r'\begin{document}'
COLOR_STR = (IMPORT_STR % ('red', 'yellow', 'green', 'blue')) + BEGIN_DOC
BLACK_STR = (IMPORT_STR % ('white', 'white', 'black', 'black')) + BEGIN_DOC

ARXIV_SRC_DIR = os.path.join(
    settings.ARXIV_DATA_OUTPUT_DIR,
    'src/')
ARXIV_MODIFIED_SRC_DIR = os.path.join(
    settings.ARXIV_DATA_OUTPUT_DIR,
    'modified_src/')
ARXIV_DIFF_DIR = os.path.join(
    settings.ARXIV_DATA_OUTPUT_DIR,
    'diffs_%ddpi/' % settings.DEFAULT_INFERENCE_DPI)
ARXIV_FIGURE_JSON_DIR = os.path.join(
    settings.ARXIV_DATA_OUTPUT_DIR,
    'figure-jsons/')
MAX_PAGES = 50

ARXIV_TAR_SRC = 's3://arxiv/src/'
ARXIV_TAR_RE = re.compile(
    ARXIV_TAR_SRC +
    'arXiv_src_(?P<year>\d\d)(?P<month>\d\d)_(?P<chunk>\d\d\d).tar'
)
ARXIV_TAR_TEMPLATE = ARXIV_TAR_SRC + 'arXiv_src_%02d%02d_%03d.tar'

PDFLATEX_TIMEOUT = 120


def doc_class_replace(match_result):
    group1 = match_result.regs[1]
    group2 = match_result.regs[2]

    group1Text = match_result.string[group1[0]: group1[1]]
    group2Text = match_result.string[group2[0]: group2[1]]
    print(group1Text, group2Text)
    if '12pt' not in group1Text:
        group1Text = group1Text + ',12pt'
    return '\documentclass[' + group1Text + ']{' + group2Text + '}'


def make_12_pt(input_text):
    return re.sub(DOC_CLASS_REGEX, doc_class_replace, input_text)


def parse_arxiv_tarname(tarname: str) -> Tuple[int, int, int]:
    match = ARXIV_TAR_RE.fullmatch(tarname)
    assert match is not None, 'Failed to match %s' % tarname
    return (
        int(match.group('year')),
        int(match.group('month')),
        int(match.group('chunk'))
    )


def generate_diffs(paper_src_dir: str,
                   dpi: int = settings.DEFAULT_INFERENCE_DPI) -> (Optional[List[str]], Optional[List[str]]):
    """
    Given the directory of a latex source file, create a modified copy of the source that includes colored boxes
    surrounding each figure and table.
    """
    paper_tex = glob.glob(paper_src_dir + '/' + '*.tex')
    if len(paper_tex) > 1:
        logging.warning('Multiple .tex files found')
        return None
    elif len(paper_tex) < 1:
        logging.warning('No .tex files found')
        return None
    texfile = paper_tex[0]
    chunk_dir, paper_id = os.path.split(paper_src_dir)
    chunk_id = os.path.basename(chunk_dir)

    # Modify latex source
    with open(texfile, 'rb') as f:
        # Some files may cause a UnicodeDecodeError if read directly as text
        # so use bs4 to fix them up
        text = bs4.UnicodeDammit(f.read()).unicode_markup
    paper_modified_src_dir = ARXIV_MODIFIED_SRC_DIR + chunk_id + '/' + paper_id
    if not os.path.isdir(paper_modified_src_dir):
        os.makedirs(paper_modified_src_dir)
    color_filename = paper_modified_src_dir + '/color.tex'
    black_filename = paper_modified_src_dir + '/black.tex'
    text = make_12_pt(text)
    with open(color_filename, 'w') as f:
        print(text.replace(BEGIN_DOC, COLOR_STR), file=f)
    with open(black_filename, 'w') as f:
        print(text.replace(BEGIN_DOC, BLACK_STR), file=f)

    result_dir = ARXIV_DIFF_DIR + chunk_id + '/' + paper_id + '/'
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    try:
        # on some PDFs, call_pdflatex doesn't raise an exception even
        # after the timeout, and instead hangs indefinitely (> 24
        # hours).
        color_pdf = figure_utils.call_pdflatex(
            src_tex=color_filename,
            src_dir=paper_src_dir,
            dest_dir=result_dir,
            timeout=PDFLATEX_TIMEOUT
        )
        black_pdf = figure_utils.call_pdflatex(
            src_tex=black_filename,
            src_dir=paper_src_dir,
            dest_dir=result_dir,
            timeout=PDFLATEX_TIMEOUT
        )
    except figure_utils.LatexException as e:
        logging.warning('Pdflatex failure: %s' % e.stdout)
        return None
    color_ims = pdf_renderer.render(color_pdf, dpi=dpi, max_pages=MAX_PAGES)
    black_ims = pdf_renderer.render(black_pdf, dpi=dpi, max_pages=MAX_PAGES)
    diff_names = []
    for (color_page, black_page) in zip(color_ims, black_ims):
        assert os.path.isfile(color_page) and os.path.isfile(black_page)
        color_page_im = imread(color_page)
        black_page_im = imread(black_page)
        assert color_page_im.shape == black_page_im.shape
        diff_page = figure_utils.im_diff(color_page_im, black_page_im)
        diff_name = result_dir + 'diff-' + os.path.basename(black_page)
        imsave(diff_name, diff_page)
        diff_names.append(diff_name)
    return diff_names, black_ims


CAPTION_LABEL_COLOR = [0, 255, 0]
CAPTION_TEXT_COLOR = [0, 0, 255]
FIGURE_BOX_COLOR = [255, 0, 0]
TABLE_BOX_COLOR = [255, 242, 0]
BACKGROUND_COLOR = [255, 255, 255]
CAPTION_OFFSET = 1


def proposal_up(full_box: BoxClass, caption_box: BoxClass) -> BoxClass:
    return BoxClass(
        x1=full_box.x1,
        y1=full_box.y1,
        x2=full_box.x2,
        y2=caption_box.y1 - CAPTION_OFFSET
    )


def proposal_down(full_box: BoxClass, caption_box: BoxClass) -> BoxClass:
    return BoxClass(
        x1=full_box.x1,
        y1=caption_box.y2 + CAPTION_OFFSET,
        x2=full_box.x2,
        y2=full_box.y2
    )


def proposal_left(full_box: BoxClass, caption_box: BoxClass) -> BoxClass:
    return BoxClass(
        x1=full_box.x1,
        y1=full_box.y1,
        x2=caption_box.x1 - CAPTION_OFFSET,
        y2=full_box.y2
    )


def proposal_right(full_box: BoxClass, caption_box: BoxClass) -> BoxClass:
    return BoxClass(
        x1=caption_box.x2 + CAPTION_OFFSET,
        y1=full_box.y1,
        x2=full_box.x2,
        y2=full_box.y2
    )


def get_figure_box(full_box: BoxClass, caption_box: BoxClass,
                   im: np.ndarray) -> Optional[BoxClass]:
    """Find the largest box inside the full figure box that doesn't overlap the caption."""
    proposals = [
        f(full_box, caption_box)
        for f in [proposal_up, proposal_down, proposal_left, proposal_right]
    ]
    proposal_areas = [p.get_area() for p in proposals]
    proposal = proposals[np.argmax(proposal_areas)]
    return proposal.crop_whitespace_edges(im)


def find_figures_and_captions(
        diff_im: np.ndarray, im: np.ndarray, page_num: int
) -> List[Figure]:
    figures = []
    all_box_mask = (
        np.logical_or(diff_im == FIGURE_BOX_COLOR, diff_im == TABLE_BOX_COLOR)
    ).all(axis=2)
    all_caption_mask = (
        np.logical_or(
            diff_im == CAPTION_LABEL_COLOR, diff_im == CAPTION_TEXT_COLOR
        )
    ).all(axis=2)
    components = measure.label(all_box_mask)
    # Component id 0 is for background
    for component_id in np.unique(components)[1:]:
        (box_ys, box_xs) = np.where(components == component_id)
        assert (len(box_ys) > 0
                )  # It was found from np.unique so it must exist somewhere
        assert (len(box_xs) > 0)
        full_box = BoxClass(
            x1=float(min(box_xs)),
            y1=float(min(box_ys)),
            x2=float(max(box_xs) + 1),
            y2=float(max(box_ys) + 1)
        )
        caption_mask = all_caption_mask.copy()
        caption_mask[:, :round(full_box.x1)] = 0
        caption_mask[:, round(full_box.x2):] = 0
        caption_mask[:round(full_box.y1), :] = 0
        caption_mask[round(full_box.y2):, :] = 0
        (cap_ys, cap_xs) = np.where(caption_mask)
        if len(cap_ys) == 0:
            continue  # Ignore boxes with no captions
        cap_box = BoxClass(
            x1=float(min(cap_xs)),
            y1=float(min(cap_ys)),
            x2=float(max(cap_xs) + 1),
            y2=float(max(cap_ys) + 1),
        )
        fig_box = get_figure_box(full_box, cap_box, im)
        if fig_box is None:
            continue
        box_color = diff_im[box_ys[0], box_xs[0], :]
        if np.all(box_color == FIGURE_BOX_COLOR):
            figure_type = 'Figure'
        else:
            assert np.all(box_color == TABLE_BOX_COLOR), print(
                'Bad box color: %s' % str(box_color)
            )
            figure_type = 'Table'
        (page_height, page_width) = diff_im.shape[:2]
        figures.append(
            Figure(
                figure_boundary=fig_box,
                caption_boundary=cap_box,
                figure_type=figure_type,
                name='',
                page=page_num,
                caption='',
                dpi=settings.DEFAULT_INFERENCE_DPI,
                page_width=page_width,
                page_height=page_height
            )
        )
    return figures


def consume_diff_generate_figures(diff) -> Optional[List[Figure]]:
    dirname = os.path.dirname(diff) + '/'
    pagenum = figure_utils.pagename_to_pagenum(diff)
    page_image_name = dirname + 'black.pdf-images/ghostscript/dpi100/black.pdf-dpi100-page%.04d.png' % (
            pagenum + 1
    )
    try:
        page_image = sp.ndimage.imread(page_image_name)
        diff_im = imread(diff)
    except Image.DecompressionBombWarning as e:
        logging.warning('Image %s too large, failed to read' % page_image_name)
        logging.warning(e)
        return None
    page_num = figure_utils.pagename_to_pagenum(page_image_name)
    figures = find_figures_and_captions(diff_im, page_image, page_num)
    return figures


def plot_bounding_box(image_path, x1, y1, x2, y2):
    im = np.array(Image.open(image_path), dtype=np.uint8)
    fig, ax = plt.subplots(1)
    ax.imshow(im)
    width = x2 - x1
    height = y2 - y1
    rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()


def augment_images(image_path, figures) -> Optional[List[Figure]]:
    # print("Running augmentation for image: {}".format(image_path))
    if len(figures) == 0:
        return figures
    image = imageio.imread(image_path)
    bbs = [ia.BoundingBox(x1=figure.figure_boundary.x1,
                          y1=figure.figure_boundary.y1,
                          x2=figure.figure_boundary.x2,
                          y2=figure.figure_boundary.y2)
           for figure in figures]
    # figg = figures[0]
    # plot_bounding_box(image_path, x1=figg.figure_boundary.x1, y1=figg.figure_boundary.y1,
    #                   x2=figg.figure_boundary.x2, y2=figg.figure_boundary.y2)
    images_aug, bbs_aug = settings.seq(images=[image], bounding_boxes=[bbs])
    imageio.imwrite(image_path, images_aug[0])
    # plot_bounding_box(image_path, x1=bbs_aug[0][0].x1, y1=bbs_aug[0][0].y1,
    #                   x2=bbs_aug[0][0].x2, y2=bbs_aug[0][0].y2)
    # print("Replaced the original image with the augmented image.")
    figures_aug = list()
    for idx, figure in enumerate(figures):
        bb = bbs_aug[0][idx]
        fig = figures[idx]
        bc = BoxClass.from_tuple((float(bb.x1), float(bb.y1), float(bb.x2), float(bb.y2)))
        fig.figure_boundary = bc
        figures_aug.append(fig)
    # print("Everything in the augmentation function complete.")
    # plot_bounding_box(image_path, x1=figures_aug[0].figure_boundary.x1, y1=figures_aug[0].figure_boundary.y1,
    #                   x2=figures_aug[0].figure_boundary.x2, y2=figures_aug[0].figure_boundary.y2)
    return figures_aug


def process_paper_tar(paper_tarname: str) -> None:
    print("------Processing paper_tarname : {}--------".format(paper_tarname))
    parts = paper_tarname.split('/')
    partition_name = parts[-2]
    paper_name = os.path.splitext(parts[-1])[0]
    result_path = os.path.join(
        ARXIV_FIGURE_JSON_DIR, partition_name, paper_name + '.json'
    )
    paper_dir = os.path.join(ARXIV_SRC_DIR, partition_name, paper_name)
    if os.path.isfile(result_path):
        return
    print('.', end='', flush=True)
    try:
        file_util.extract_tarfile(paper_tarname, paper_dir)
    except tarfile.ReadError:
        logging.debug('File %s is not a tar' % paper_tarname)
        return
    try:
        diffs, black_ims_paths = generate_diffs(paper_dir)
    except TypeError:
        return
    if diffs is None:
        return
    figures_by_page = dict()
    for idx, diff in enumerate(diffs):
        figures = consume_diff_generate_figures(diff)
        if figures is None:
            continue
        try:
            figures = augment_images(black_ims_paths[idx], figures)
        except Exception as e:
            print("Error augmenting images for image path: {}. Exception message: {}".format(black_ims_paths[idx], e))
        page_name = os.path.dirname(diff) + '/' + diff[diff.find('black.pdf-'):]
        figures_by_page[page_name] = figures
    file_util.safe_makedirs(os.path.dirname(result_path))
    file_util.write_json_atomic(
        result_path,
        config.JsonSerializable.serialize(figures_by_page),
        sort_keys=True
    )


def process_paper_tar_with_timeout(paper_tarname: str) -> None:
    p = ThreadPool(processes=1)
    result = p.apply_async(process_paper_tar, (paper_tarname,))
    try:
        out = result.get(timeout=100)  # Wait timeout seconds for func to complete.
        return out
    except multiprocessing.TimeoutError:
        print("Aborting due to timeout")
        p.terminate()
        raise


def download_and_extract_tar(
        tarname: str, extract_dir: str, n_attempts: int = 100
) -> None:
    print('.', end='', flush=True)
    logging.info('Downloading %s' % tarname)
    for attempt in range(n_attempts):
        try:
            cached_file = file_util.cache_file_2(tarname, cache_dir=settings.ARXIV_DATA_CACHE_DIR)
            break
        except FileNotFoundError:
            if attempt == n_attempts - 1:
                raise
            logging.exception('Download failed, retrying')
            time.sleep(10)
    # logging.info("Proceeding to extract tar file: {}".format(cached_file))
    # file_util.extract_tarfile(cached_file, extract_dir)
    # os.remove(cached_file)


def run_on_all() -> None:
    Image.MAX_IMAGE_PIXELS = int(1e8)  # Don't render very large PDFs.
    Image.warnings.simplefilter('error', Image.DecompressionBombWarning)
    # tarnames = [
    #     tarname for tarname in file_util.iterate_s3_files(ARXIV_TAR_SRC)
    #     if os.path.splitext(tarname)[1] == '.tar'
    # ]
    # import json
    # import random
    # all_tarnames = json.load(open('all_tarnames.json'))
    # tarnames = random.choices(all_tarnames, k=31)
    # print(tarnames)
    tarnames = [
        "s3://arxiv/src/arXiv_src_0001_001.tar",
        "s3://arxiv/src/arXiv_src_0002_001.tar",
        "s3://arxiv/src/arXiv_src_0003_001.tar",
        "s3://arxiv/src/arXiv_src_0004_001.tar",
        "s3://arxiv/src/arXiv_src_0005_001.tar",
        "s3://arxiv/src/arXiv_src_0006_001.tar",
        "s3://arxiv/src/arXiv_src_0007_001.tar",
        "s3://arxiv/src/arXiv_src_0008_001.tar",
        "s3://arxiv/src/arXiv_src_0009_001.tar",
        "s3://arxiv/src/arXiv_src_0010_001.tar",
        "s3://arxiv/src/arXiv_src_0011_001.tar",
        "s3://arxiv/src/arXiv_src_0012_001.tar",
        "s3://arxiv/src/arXiv_src_0101_001.tar",
        "s3://arxiv/src/arXiv_src_0102_001.tar",
        "s3://arxiv/src/arXiv_src_0103_001.tar",
        "s3://arxiv/src/arXiv_src_0104_001.tar",
        "s3://arxiv/src/arXiv_src_0105_001.tar",
        "s3://arxiv/src/arXiv_src_0106_001.tar",
        "s3://arxiv/src/arXiv_src_0107_001.tar",
        "s3://arxiv/src/arXiv_src_0108_001.tar",
        "s3://arxiv/src/arXiv_src_0109_001.tar",
        "s3://arxiv/src/arXiv_src_0110_001.tar",
        "s3://arxiv/src/arXiv_src_0111_001.tar",
        "s3://arxiv/src/arXiv_src_0112_001.tar",
        "s3://arxiv/src/arXiv_src_0201_001.tar",
        "s3://arxiv/src/arXiv_src_0202_001.tar",
        "s3://arxiv/src/arXiv_src_0203_001.tar",
        "s3://arxiv/src/arXiv_src_0204_001.tar",
        "s3://arxiv/src/arXiv_src_0205_001.tar",
        "s3://arxiv/src/arXiv_src_0206_001.tar",
        "s3://arxiv/src/arXiv_src_0207_001.tar",
        "s3://arxiv/src/arXiv_src_0208_001.tar",
        "s3://arxiv/src/arXiv_src_0209_001.tar",
        "s3://arxiv/src/arXiv_src_0210_001.tar",
        "s3://arxiv/src/arXiv_src_0211_001.tar",
        "s3://arxiv/src/arXiv_src_0212_001.tar",
        "s3://arxiv/src/arXiv_src_0301_001.tar",
        "s3://arxiv/src/arXiv_src_0302_001.tar",
        "s3://arxiv/src/arXiv_src_0303_001.tar",
        "s3://arxiv/src/arXiv_src_0304_001.tar",
        "s3://arxiv/src/arXiv_src_0305_001.tar",
        "s3://arxiv/src/arXiv_src_0306_001.tar",
        "s3://arxiv/src/arXiv_src_0307_001.tar",
        "s3://arxiv/src/arXiv_src_0308_001.tar",
        "s3://arxiv/src/arXiv_src_0309_001.tar",
        "s3://arxiv/src/arXiv_src_0310_001.tar",
        "s3://arxiv/src/arXiv_src_0311_001.tar",
        "s3://arxiv/src/arXiv_src_0312_001.tar",
        "s3://arxiv/src/arXiv_src_0401_001.tar",
        "s3://arxiv/src/arXiv_src_0402_001.tar",
        "s3://arxiv/src/arXiv_src_0403_001.tar",
        "s3://arxiv/src/arXiv_src_0404_001.tar",
        "s3://arxiv/src/arXiv_src_0405_001.tar",
        "s3://arxiv/src/arXiv_src_0406_001.tar",
        "s3://arxiv/src/arXiv_src_0406_002.tar",
        "s3://arxiv/src/arXiv_src_0407_001.tar",
        "s3://arxiv/src/arXiv_src_0407_002.tar",
        "s3://arxiv/src/arXiv_src_0408_001.tar",
        "s3://arxiv/src/arXiv_src_0409_001.tar",
        "s3://arxiv/src/arXiv_src_0409_002.tar",
        "s3://arxiv/src/arXiv_src_0410_001.tar",
        "s3://arxiv/src/arXiv_src_0410_002.tar",
        "s3://arxiv/src/arXiv_src_0411_001.tar",
        "s3://arxiv/src/arXiv_src_0411_002.tar",
        "s3://arxiv/src/arXiv_src_0412_001.tar",
        "s3://arxiv/src/arXiv_src_0412_002.tar",
        "s3://arxiv/src/arXiv_src_0501_001.tar",
        "s3://arxiv/src/arXiv_src_0501_002.tar",
        "s3://arxiv/src/arXiv_src_0502_001.tar",
        "s3://arxiv/src/arXiv_src_0502_002.tar",
        "s3://arxiv/src/arXiv_src_0503_001.tar",
        "s3://arxiv/src/arXiv_src_0503_002.tar",
        "s3://arxiv/src/arXiv_src_0504_001.tar",
        "s3://arxiv/src/arXiv_src_0504_002.tar",
        "s3://arxiv/src/arXiv_src_0505_001.tar",
        "s3://arxiv/src/arXiv_src_0505_002.tar",
        "s3://arxiv/src/arXiv_src_0506_001.tar",
        "s3://arxiv/src/arXiv_src_0506_002.tar",
        "s3://arxiv/src/arXiv_src_0507_001.tar",
        "s3://arxiv/src/arXiv_src_0507_002.tar",
        "s3://arxiv/src/arXiv_src_0508_001.tar",
        "s3://arxiv/src/arXiv_src_0508_002.tar",
        "s3://arxiv/src/arXiv_src_0509_001.tar",
        "s3://arxiv/src/arXiv_src_0509_002.tar",
        "s3://arxiv/src/arXiv_src_0510_001.tar",
        "s3://arxiv/src/arXiv_src_0510_002.tar",
        "s3://arxiv/src/arXiv_src_0511_001.tar",
        "s3://arxiv/src/arXiv_src_0511_002.tar",
        "s3://arxiv/src/arXiv_src_0512_001.tar",
        "s3://arxiv/src/arXiv_src_0512_002.tar",
        "s3://arxiv/src/arXiv_src_0601_001.tar",
        "s3://arxiv/src/arXiv_src_0601_002.tar",
        "s3://arxiv/src/arXiv_src_0602_001.tar",
        "s3://arxiv/src/arXiv_src_0602_002.tar",
        "s3://arxiv/src/arXiv_src_0603_001.tar",
        "s3://arxiv/src/arXiv_src_0603_002.tar",
        "s3://arxiv/src/arXiv_src_0604_001.tar",
        "s3://arxiv/src/arXiv_src_0604_002.tar",
        "s3://arxiv/src/arXiv_src_0605_001.tar",
        "s3://arxiv/src/arXiv_src_0605_002.tar",
        "s3://arxiv/src/arXiv_src_0606_001.tar",
        "s3://arxiv/src/arXiv_src_0606_002.tar",
        "s3://arxiv/src/arXiv_src_0607_001.tar",
        "s3://arxiv/src/arXiv_src_0607_002.tar",
        "s3://arxiv/src/arXiv_src_0608_001.tar",
        "s3://arxiv/src/arXiv_src_0608_002.tar",
        "s3://arxiv/src/arXiv_src_0609_001.tar",
        "s3://arxiv/src/arXiv_src_0609_002.tar",
        "s3://arxiv/src/arXiv_src_0610_001.tar",
        "s3://arxiv/src/arXiv_src_0610_002.tar",
        "s3://arxiv/src/arXiv_src_0610_003.tar",
        "s3://arxiv/src/arXiv_src_0611_001.tar",
        "s3://arxiv/src/arXiv_src_0611_002.tar",
        "s3://arxiv/src/arXiv_src_0612_001.tar",
        "s3://arxiv/src/arXiv_src_0612_002.tar",
        "s3://arxiv/src/arXiv_src_0701_001.tar",
        "s3://arxiv/src/arXiv_src_0701_002.tar",
        "s3://arxiv/src/arXiv_src_0702_001.tar",
        "s3://arxiv/src/arXiv_src_0702_002.tar",
        "s3://arxiv/src/arXiv_src_0703_001.tar",
        "s3://arxiv/src/arXiv_src_0703_002.tar",
        "s3://arxiv/src/arXiv_src_0704_001.tar",
        "s3://arxiv/src/arXiv_src_0704_002.tar",
        "s3://arxiv/src/arXiv_src_0705_001.tar",
        "s3://arxiv/src/arXiv_src_0705_002.tar",
        "s3://arxiv/src/arXiv_src_0706_001.tar",
        "s3://arxiv/src/arXiv_src_0706_002.tar",
        "s3://arxiv/src/arXiv_src_0707_001.tar",
        "s3://arxiv/src/arXiv_src_0707_002.tar",
        "s3://arxiv/src/arXiv_src_0708_001.tar",
        "s3://arxiv/src/arXiv_src_0708_002.tar",
        "s3://arxiv/src/arXiv_src_0709_001.tar",
        "s3://arxiv/src/arXiv_src_0709_002.tar",
        "s3://arxiv/src/arXiv_src_0709_003.tar",
        "s3://arxiv/src/arXiv_src_0710_001.tar",
        "s3://arxiv/src/arXiv_src_0710_002.tar",
        "s3://arxiv/src/arXiv_src_0710_003.tar",
        "s3://arxiv/src/arXiv_src_0711_001.tar",
        "s3://arxiv/src/arXiv_src_0711_002.tar",
        "s3://arxiv/src/arXiv_src_0711_003.tar",
        "s3://arxiv/src/arXiv_src_0712_001.tar",
        "s3://arxiv/src/arXiv_src_0712_002.tar",
        "s3://arxiv/src/arXiv_src_0712_003.tar",
        "s3://arxiv/src/arXiv_src_0801_001.tar",
        "s3://arxiv/src/arXiv_src_0801_002.tar",
        "s3://arxiv/src/arXiv_src_0801_003.tar",
        "s3://arxiv/src/arXiv_src_0802_001.tar",
        "s3://arxiv/src/arXiv_src_0802_002.tar",
        "s3://arxiv/src/arXiv_src_0802_003.tar",
        "s3://arxiv/src/arXiv_src_0803_001.tar",
        "s3://arxiv/src/arXiv_src_0803_002.tar",
        "s3://arxiv/src/arXiv_src_0803_003.tar",
        "s3://arxiv/src/arXiv_src_0804_001.tar",
        "s3://arxiv/src/arXiv_src_0804_002.tar",
        "s3://arxiv/src/arXiv_src_0804_003.tar",
        "s3://arxiv/src/arXiv_src_0805_001.tar",
        "s3://arxiv/src/arXiv_src_0805_002.tar",
        "s3://arxiv/src/arXiv_src_0805_003.tar",
        "s3://arxiv/src/arXiv_src_0806_001.tar",
        "s3://arxiv/src/arXiv_src_0806_002.tar",
        "s3://arxiv/src/arXiv_src_0806_003.tar",
        "s3://arxiv/src/arXiv_src_0807_001.tar",
        "s3://arxiv/src/arXiv_src_0807_002.tar",
        "s3://arxiv/src/arXiv_src_0807_003.tar",
        "s3://arxiv/src/arXiv_src_0807_004.tar",
        "s3://arxiv/src/arXiv_src_0808_001.tar",
        "s3://arxiv/src/arXiv_src_0808_002.tar",
        "s3://arxiv/src/arXiv_src_0808_003.tar",
        "s3://arxiv/src/arXiv_src_0809_001.tar",
        "s3://arxiv/src/arXiv_src_0809_002.tar",
        "s3://arxiv/src/arXiv_src_0809_003.tar",
        "s3://arxiv/src/arXiv_src_0809_004.tar",
        "s3://arxiv/src/arXiv_src_0810_001.tar",
        "s3://arxiv/src/arXiv_src_0810_002.tar",
        "s3://arxiv/src/arXiv_src_0810_003.tar",
        "s3://arxiv/src/arXiv_src_0810_004.tar",
        "s3://arxiv/src/arXiv_src_0810_005.tar",
        "s3://arxiv/src/arXiv_src_0811_001.tar",
        "s3://arxiv/src/arXiv_src_0811_002.tar",
        "s3://arxiv/src/arXiv_src_0811_003.tar",
        "s3://arxiv/src/arXiv_src_0811_004.tar",
        "s3://arxiv/src/arXiv_src_0812_001.tar",
        "s3://arxiv/src/arXiv_src_0812_002.tar",
        "s3://arxiv/src/arXiv_src_0812_003.tar",
        "s3://arxiv/src/arXiv_src_0812_004.tar",
        "s3://arxiv/src/arXiv_src_0901_001.tar",
        "s3://arxiv/src/arXiv_src_0901_002.tar",
        "s3://arxiv/src/arXiv_src_0901_003.tar",
        "s3://arxiv/src/arXiv_src_0901_004.tar",
        "s3://arxiv/src/arXiv_src_0902_001.tar",
        "s3://arxiv/src/arXiv_src_0902_002.tar",
        "s3://arxiv/src/arXiv_src_0902_003.tar",
        "s3://arxiv/src/arXiv_src_0902_004.tar",
        "s3://arxiv/src/arXiv_src_0903_001.tar",
        "s3://arxiv/src/arXiv_src_0903_002.tar",
        "s3://arxiv/src/arXiv_src_0903_003.tar",
        "s3://arxiv/src/arXiv_src_0903_004.tar",
        "s3://arxiv/src/arXiv_src_0903_005.tar",
        "s3://arxiv/src/arXiv_src_0904_001.tar",
        "s3://arxiv/src/arXiv_src_0904_002.tar",
        "s3://arxiv/src/arXiv_src_0904_003.tar",
        "s3://arxiv/src/arXiv_src_0904_004.tar",
        "s3://arxiv/src/arXiv_src_0905_001.tar",
        "s3://arxiv/src/arXiv_src_0905_002.tar",
        "s3://arxiv/src/arXiv_src_0905_003.tar",
        "s3://arxiv/src/arXiv_src_0905_004.tar",
        "s3://arxiv/src/arXiv_src_0906_001.tar",
        "s3://arxiv/src/arXiv_src_0906_002.tar",
        "s3://arxiv/src/arXiv_src_0906_003.tar",
        "s3://arxiv/src/arXiv_src_0906_004.tar",
        "s3://arxiv/src/arXiv_src_0906_005.tar",
        "s3://arxiv/src/arXiv_src_0907_001.tar",
        "s3://arxiv/src/arXiv_src_0907_002.tar",
        "s3://arxiv/src/arXiv_src_0907_003.tar",
        "s3://arxiv/src/arXiv_src_0907_004.tar",
        "s3://arxiv/src/arXiv_src_0907_005.tar",
        "s3://arxiv/src/arXiv_src_0908_001.tar",
        "s3://arxiv/src/arXiv_src_0908_002.tar",
        "s3://arxiv/src/arXiv_src_0908_003.tar",
        "s3://arxiv/src/arXiv_src_0908_004.tar",
        "s3://arxiv/src/arXiv_src_0909_001.tar",
        "s3://arxiv/src/arXiv_src_0909_002.tar",
        "s3://arxiv/src/arXiv_src_0909_003.tar",
        "s3://arxiv/src/arXiv_src_0909_004.tar",
        "s3://arxiv/src/arXiv_src_0909_005.tar",
        "s3://arxiv/src/arXiv_src_0910_001.tar",
        "s3://arxiv/src/arXiv_src_0910_002.tar",
        "s3://arxiv/src/arXiv_src_0910_003.tar",
        "s3://arxiv/src/arXiv_src_0910_004.tar",
        "s3://arxiv/src/arXiv_src_0910_005.tar",
        "s3://arxiv/src/arXiv_src_0911_001.tar",
        "s3://arxiv/src/arXiv_src_0911_002.tar",
        "s3://arxiv/src/arXiv_src_0911_003.tar",
        "s3://arxiv/src/arXiv_src_0911_004.tar",
        "s3://arxiv/src/arXiv_src_0911_005.tar",
        "s3://arxiv/src/arXiv_src_0912_001.tar",
        "s3://arxiv/src/arXiv_src_0912_002.tar",
        "s3://arxiv/src/arXiv_src_0912_003.tar",
        "s3://arxiv/src/arXiv_src_0912_004.tar",
        "s3://arxiv/src/arXiv_src_0912_005.tar",
        "s3://arxiv/src/arXiv_src_1001_001.tar",
        "s3://arxiv/src/arXiv_src_1001_002.tar",
        "s3://arxiv/src/arXiv_src_1001_003.tar",
        "s3://arxiv/src/arXiv_src_1001_004.tar",
        "s3://arxiv/src/arXiv_src_1001_005.tar",
        "s3://arxiv/src/arXiv_src_1002_001.tar",
        "s3://arxiv/src/arXiv_src_1002_002.tar",
        "s3://arxiv/src/arXiv_src_1002_003.tar",
        "s3://arxiv/src/arXiv_src_1002_004.tar",
        "s3://arxiv/src/arXiv_src_1002_005.tar",
        "s3://arxiv/src/arXiv_src_1003_001.tar",
        "s3://arxiv/src/arXiv_src_1003_002.tar",
        "s3://arxiv/src/arXiv_src_1003_003.tar",
        "s3://arxiv/src/arXiv_src_1003_004.tar",
        "s3://arxiv/src/arXiv_src_1003_005.tar",
        "s3://arxiv/src/arXiv_src_1003_006.tar",
        "s3://arxiv/src/arXiv_src_1004_001.tar",
        "s3://arxiv/src/arXiv_src_1004_002.tar",
        "s3://arxiv/src/arXiv_src_1004_003.tar",
        "s3://arxiv/src/arXiv_src_1004_004.tar",
        "s3://arxiv/src/arXiv_src_1004_005.tar",
        "s3://arxiv/src/arXiv_src_1004_006.tar",
        "s3://arxiv/src/arXiv_src_1005_001.tar",
        "s3://arxiv/src/arXiv_src_1005_002.tar",
        "s3://arxiv/src/arXiv_src_1005_003.tar",
        "s3://arxiv/src/arXiv_src_1005_004.tar",
        "s3://arxiv/src/arXiv_src_1005_005.tar",
        "s3://arxiv/src/arXiv_src_1005_006.tar",
        "s3://arxiv/src/arXiv_src_1006_001.tar",
        "s3://arxiv/src/arXiv_src_1006_002.tar",
        "s3://arxiv/src/arXiv_src_1006_003.tar",
        "s3://arxiv/src/arXiv_src_1006_004.tar",
        "s3://arxiv/src/arXiv_src_1006_005.tar",
        "s3://arxiv/src/arXiv_src_1006_006.tar",
        "s3://arxiv/src/arXiv_src_1007_001.tar",
        "s3://arxiv/src/arXiv_src_1007_002.tar",
        "s3://arxiv/src/arXiv_src_1007_003.tar",
        "s3://arxiv/src/arXiv_src_1007_004.tar",
        "s3://arxiv/src/arXiv_src_1007_005.tar",
        "s3://arxiv/src/arXiv_src_1007_006.tar",
        "s3://arxiv/src/arXiv_src_1008_001.tar",
        "s3://arxiv/src/arXiv_src_1008_002.tar",
        "s3://arxiv/src/arXiv_src_1008_003.tar",
        "s3://arxiv/src/arXiv_src_1008_004.tar",
        "s3://arxiv/src/arXiv_src_1008_005.tar",
        "s3://arxiv/src/arXiv_src_1008_006.tar",
        "s3://arxiv/src/arXiv_src_1009_001.tar",
        "s3://arxiv/src/arXiv_src_1009_002.tar",
        "s3://arxiv/src/arXiv_src_1009_003.tar",
        "s3://arxiv/src/arXiv_src_1009_004.tar",
        "s3://arxiv/src/arXiv_src_1009_005.tar",
        "s3://arxiv/src/arXiv_src_1009_006.tar",
        "s3://arxiv/src/arXiv_src_1009_007.tar",
        "s3://arxiv/src/arXiv_src_1009_008.tar",
        "s3://arxiv/src/arXiv_src_1010_001.tar",
        "s3://arxiv/src/arXiv_src_1010_002.tar",
        "s3://arxiv/src/arXiv_src_1010_003.tar",
        "s3://arxiv/src/arXiv_src_1010_004.tar",
        "s3://arxiv/src/arXiv_src_1010_005.tar",
        "s3://arxiv/src/arXiv_src_1010_006.tar",
        "s3://arxiv/src/arXiv_src_1010_007.tar",
        "s3://arxiv/src/arXiv_src_1010_008.tar",
        "s3://arxiv/src/arXiv_src_1010_009.tar",
        "s3://arxiv/src/arXiv_src_1010_010.tar",
        "s3://arxiv/src/arXiv_src_1010_011.tar",
        "s3://arxiv/src/arXiv_src_1010_012.tar",
        "s3://arxiv/src/arXiv_src_1010_013.tar",
        "s3://arxiv/src/arXiv_src_1010_014.tar",
        "s3://arxiv/src/arXiv_src_1010_015.tar",
        "s3://arxiv/src/arXiv_src_1011_001.tar",
        "s3://arxiv/src/arXiv_src_1011_002.tar",
        "s3://arxiv/src/arXiv_src_1011_003.tar",
        "s3://arxiv/src/arXiv_src_1011_004.tar",
        "s3://arxiv/src/arXiv_src_1011_005.tar",
        "s3://arxiv/src/arXiv_src_1011_006.tar",
        "s3://arxiv/src/arXiv_src_1011_007.tar",
        "s3://arxiv/src/arXiv_src_1012_001.tar",
        "s3://arxiv/src/arXiv_src_1012_002.tar",
        "s3://arxiv/src/arXiv_src_1012_003.tar",
        "s3://arxiv/src/arXiv_src_1012_004.tar",
        "s3://arxiv/src/arXiv_src_1012_005.tar",
        "s3://arxiv/src/arXiv_src_1012_006.tar",
        "s3://arxiv/src/arXiv_src_1012_007.tar",
        "s3://arxiv/src/arXiv_src_1101_001.tar",
        "s3://arxiv/src/arXiv_src_1101_002.tar",
        "s3://arxiv/src/arXiv_src_1101_003.tar",
        "s3://arxiv/src/arXiv_src_1101_004.tar",
        "s3://arxiv/src/arXiv_src_1101_005.tar",
        "s3://arxiv/src/arXiv_src_1101_006.tar",
        "s3://arxiv/src/arXiv_src_1101_007.tar",
        "s3://arxiv/src/arXiv_src_1102_001.tar",
        "s3://arxiv/src/arXiv_src_1102_002.tar",
        "s3://arxiv/src/arXiv_src_1102_003.tar",
        "s3://arxiv/src/arXiv_src_1102_004.tar",
        "s3://arxiv/src/arXiv_src_1102_005.tar",
        "s3://arxiv/src/arXiv_src_1102_006.tar",
        "s3://arxiv/src/arXiv_src_1102_007.tar",
        "s3://arxiv/src/arXiv_src_1103_001.tar",
        "s3://arxiv/src/arXiv_src_1103_002.tar",
        "s3://arxiv/src/arXiv_src_1103_003.tar",
        "s3://arxiv/src/arXiv_src_1103_004.tar",
        "s3://arxiv/src/arXiv_src_1103_005.tar",
        "s3://arxiv/src/arXiv_src_1103_006.tar",
        "s3://arxiv/src/arXiv_src_1103_007.tar",
        "s3://arxiv/src/arXiv_src_1104_001.tar",
        "s3://arxiv/src/arXiv_src_1104_002.tar",
        "s3://arxiv/src/arXiv_src_1104_003.tar",
        "s3://arxiv/src/arXiv_src_1104_004.tar",
        "s3://arxiv/src/arXiv_src_1104_005.tar",
        "s3://arxiv/src/arXiv_src_1104_006.tar",
        "s3://arxiv/src/arXiv_src_1104_007.tar",
        "s3://arxiv/src/arXiv_src_1105_001.tar",
        "s3://arxiv/src/arXiv_src_1105_002.tar",
        "s3://arxiv/src/arXiv_src_1105_003.tar",
        "s3://arxiv/src/arXiv_src_1105_004.tar",
        "s3://arxiv/src/arXiv_src_1105_005.tar",
        "s3://arxiv/src/arXiv_src_1105_006.tar",
        "s3://arxiv/src/arXiv_src_1105_007.tar",
        "s3://arxiv/src/arXiv_src_1106_001.tar",
        "s3://arxiv/src/arXiv_src_1106_002.tar",
        "s3://arxiv/src/arXiv_src_1106_003.tar",
        "s3://arxiv/src/arXiv_src_1106_004.tar",
        "s3://arxiv/src/arXiv_src_1106_005.tar",
        "s3://arxiv/src/arXiv_src_1106_006.tar",
        "s3://arxiv/src/arXiv_src_1106_007.tar",
        "s3://arxiv/src/arXiv_src_1107_001.tar",
        "s3://arxiv/src/arXiv_src_1107_002.tar",
        "s3://arxiv/src/arXiv_src_1107_003.tar",
        "s3://arxiv/src/arXiv_src_1107_004.tar",
        "s3://arxiv/src/arXiv_src_1107_005.tar",
        "s3://arxiv/src/arXiv_src_1107_006.tar",
        "s3://arxiv/src/arXiv_src_1107_007.tar",
        "s3://arxiv/src/arXiv_src_1108_001.tar",
        "s3://arxiv/src/arXiv_src_1108_002.tar",
        "s3://arxiv/src/arXiv_src_1108_003.tar",
        "s3://arxiv/src/arXiv_src_1108_004.tar",
        "s3://arxiv/src/arXiv_src_1108_005.tar",
        "s3://arxiv/src/arXiv_src_1108_006.tar",
        "s3://arxiv/src/arXiv_src_1108_007.tar",
        "s3://arxiv/src/arXiv_src_1108_008.tar",
        "s3://arxiv/src/arXiv_src_1109_001.tar",
        "s3://arxiv/src/arXiv_src_1109_002.tar",
        "s3://arxiv/src/arXiv_src_1109_003.tar",
        "s3://arxiv/src/arXiv_src_1109_004.tar",
        "s3://arxiv/src/arXiv_src_1109_005.tar",
        "s3://arxiv/src/arXiv_src_1109_006.tar",
        "s3://arxiv/src/arXiv_src_1109_007.tar",
        "s3://arxiv/src/arXiv_src_1109_008.tar",
        "s3://arxiv/src/arXiv_src_1109_009.tar",
        "s3://arxiv/src/arXiv_src_1110_001.tar",
        "s3://arxiv/src/arXiv_src_1110_002.tar",
        "s3://arxiv/src/arXiv_src_1110_003.tar",
        "s3://arxiv/src/arXiv_src_1110_004.tar",
        "s3://arxiv/src/arXiv_src_1110_005.tar",
        "s3://arxiv/src/arXiv_src_1110_006.tar",
        "s3://arxiv/src/arXiv_src_1110_007.tar",
        "s3://arxiv/src/arXiv_src_1110_008.tar",
        "s3://arxiv/src/arXiv_src_1110_009.tar",
        "s3://arxiv/src/arXiv_src_1110_010.tar",
        "s3://arxiv/src/arXiv_src_1110_011.tar",
        "s3://arxiv/src/arXiv_src_1110_012.tar",
        "s3://arxiv/src/arXiv_src_1110_013.tar",
        "s3://arxiv/src/arXiv_src_1110_014.tar",
        "s3://arxiv/src/arXiv_src_1110_015.tar",
        "s3://arxiv/src/arXiv_src_1110_016.tar",
        "s3://arxiv/src/arXiv_src_1111_001.tar",
        "s3://arxiv/src/arXiv_src_1111_002.tar",
        "s3://arxiv/src/arXiv_src_1111_003.tar",
        "s3://arxiv/src/arXiv_src_1111_004.tar",
        "s3://arxiv/src/arXiv_src_1111_005.tar",
        "s3://arxiv/src/arXiv_src_1111_006.tar",
        "s3://arxiv/src/arXiv_src_1111_007.tar",
        "s3://arxiv/src/arXiv_src_1111_008.tar",
        "s3://arxiv/src/arXiv_src_1111_009.tar",
        "s3://arxiv/src/arXiv_src_1112_001.tar",
        "s3://arxiv/src/arXiv_src_1112_002.tar",
        "s3://arxiv/src/arXiv_src_1112_003.tar",
        "s3://arxiv/src/arXiv_src_1112_004.tar",
        "s3://arxiv/src/arXiv_src_1112_005.tar",
        "s3://arxiv/src/arXiv_src_1112_006.tar",
        "s3://arxiv/src/arXiv_src_1112_007.tar",
        "s3://arxiv/src/arXiv_src_1112_008.tar",
        "s3://arxiv/src/arXiv_src_1201_001.tar",
        "s3://arxiv/src/arXiv_src_1201_002.tar",
        "s3://arxiv/src/arXiv_src_1201_003.tar",
        "s3://arxiv/src/arXiv_src_1201_004.tar",
        "s3://arxiv/src/arXiv_src_1201_005.tar",
        "s3://arxiv/src/arXiv_src_1201_006.tar",
        "s3://arxiv/src/arXiv_src_1201_007.tar",
        "s3://arxiv/src/arXiv_src_1201_008.tar",
        "s3://arxiv/src/arXiv_src_1201_009.tar",
        "s3://arxiv/src/arXiv_src_1202_001.tar",
        "s3://arxiv/src/arXiv_src_1202_002.tar",
        "s3://arxiv/src/arXiv_src_1202_003.tar",
        "s3://arxiv/src/arXiv_src_1202_004.tar",
        "s3://arxiv/src/arXiv_src_1202_005.tar",
        "s3://arxiv/src/arXiv_src_1202_006.tar",
        "s3://arxiv/src/arXiv_src_1202_007.tar",
        "s3://arxiv/src/arXiv_src_1202_008.tar",
        "s3://arxiv/src/arXiv_src_1202_009.tar",
        "s3://arxiv/src/arXiv_src_1203_001.tar",
        "s3://arxiv/src/arXiv_src_1203_002.tar",
        "s3://arxiv/src/arXiv_src_1203_003.tar",
        "s3://arxiv/src/arXiv_src_1203_004.tar",
        "s3://arxiv/src/arXiv_src_1203_005.tar",
        "s3://arxiv/src/arXiv_src_1203_006.tar",
        "s3://arxiv/src/arXiv_src_1203_007.tar",
        "s3://arxiv/src/arXiv_src_1203_008.tar",
        "s3://arxiv/src/arXiv_src_1203_009.tar",
        "s3://arxiv/src/arXiv_src_1204_001.tar",
        "s3://arxiv/src/arXiv_src_1204_002.tar",
        "s3://arxiv/src/arXiv_src_1204_003.tar",
        "s3://arxiv/src/arXiv_src_1204_004.tar",
        "s3://arxiv/src/arXiv_src_1204_005.tar",
        "s3://arxiv/src/arXiv_src_1204_006.tar",
        "s3://arxiv/src/arXiv_src_1204_007.tar",
        "s3://arxiv/src/arXiv_src_1204_008.tar",
        "s3://arxiv/src/arXiv_src_1204_009.tar",
        "s3://arxiv/src/arXiv_src_1205_001.tar",
        "s3://arxiv/src/arXiv_src_1205_002.tar",
        "s3://arxiv/src/arXiv_src_1205_003.tar",
        "s3://arxiv/src/arXiv_src_1205_004.tar",
        "s3://arxiv/src/arXiv_src_1205_005.tar",
        "s3://arxiv/src/arXiv_src_1205_006.tar",
        "s3://arxiv/src/arXiv_src_1205_007.tar",
        "s3://arxiv/src/arXiv_src_1205_008.tar",
        "s3://arxiv/src/arXiv_src_1205_009.tar",
        "s3://arxiv/src/arXiv_src_1205_010.tar",
        "s3://arxiv/src/arXiv_src_1206_001.tar",
        "s3://arxiv/src/arXiv_src_1206_002.tar",
        "s3://arxiv/src/arXiv_src_1206_003.tar",
        "s3://arxiv/src/arXiv_src_1206_004.tar",
        "s3://arxiv/src/arXiv_src_1206_005.tar",
        "s3://arxiv/src/arXiv_src_1206_006.tar",
        "s3://arxiv/src/arXiv_src_1206_007.tar",
        "s3://arxiv/src/arXiv_src_1206_008.tar",
        "s3://arxiv/src/arXiv_src_1206_009.tar",
        "s3://arxiv/src/arXiv_src_1206_010.tar",
        "s3://arxiv/src/arXiv_src_1207_001.tar",
        "s3://arxiv/src/arXiv_src_1207_002.tar",
        "s3://arxiv/src/arXiv_src_1207_003.tar",
        "s3://arxiv/src/arXiv_src_1207_004.tar",
        "s3://arxiv/src/arXiv_src_1207_005.tar",
        "s3://arxiv/src/arXiv_src_1207_006.tar",
        "s3://arxiv/src/arXiv_src_1207_007.tar",
        "s3://arxiv/src/arXiv_src_1207_008.tar",
        "s3://arxiv/src/arXiv_src_1207_009.tar",
        "s3://arxiv/src/arXiv_src_1207_010.tar",
        "s3://arxiv/src/arXiv_src_1207_011.tar",
        "s3://arxiv/src/arXiv_src_1208_001.tar",
        "s3://arxiv/src/arXiv_src_1208_002.tar",
        "s3://arxiv/src/arXiv_src_1208_003.tar",
        "s3://arxiv/src/arXiv_src_1208_004.tar",
        "s3://arxiv/src/arXiv_src_1208_005.tar",
        "s3://arxiv/src/arXiv_src_1208_006.tar",
        "s3://arxiv/src/arXiv_src_1208_007.tar",
        "s3://arxiv/src/arXiv_src_1208_008.tar",
        "s3://arxiv/src/arXiv_src_1208_009.tar",
        "s3://arxiv/src/arXiv_src_1208_010.tar",
        "s3://arxiv/src/arXiv_src_1209_001.tar",
        "s3://arxiv/src/arXiv_src_1209_002.tar",
        "s3://arxiv/src/arXiv_src_1209_003.tar",
        "s3://arxiv/src/arXiv_src_1209_004.tar",
        "s3://arxiv/src/arXiv_src_1209_005.tar",
        "s3://arxiv/src/arXiv_src_1209_006.tar",
        "s3://arxiv/src/arXiv_src_1209_007.tar",
        "s3://arxiv/src/arXiv_src_1209_008.tar",
        "s3://arxiv/src/arXiv_src_1209_009.tar",
        "s3://arxiv/src/arXiv_src_1209_010.tar",
        "s3://arxiv/src/arXiv_src_1210_001.tar",
        "s3://arxiv/src/arXiv_src_1210_002.tar",
        "s3://arxiv/src/arXiv_src_1210_003.tar",
        "s3://arxiv/src/arXiv_src_1210_004.tar",
        "s3://arxiv/src/arXiv_src_1210_005.tar",
        "s3://arxiv/src/arXiv_src_1210_006.tar",
        "s3://arxiv/src/arXiv_src_1210_007.tar",
        "s3://arxiv/src/arXiv_src_1210_008.tar",
        "s3://arxiv/src/arXiv_src_1210_009.tar",
        "s3://arxiv/src/arXiv_src_1210_010.tar",
        "s3://arxiv/src/arXiv_src_1210_011.tar",
        "s3://arxiv/src/arXiv_src_1210_012.tar",
        "s3://arxiv/src/arXiv_src_1210_013.tar",
        "s3://arxiv/src/arXiv_src_1210_014.tar",
        "s3://arxiv/src/arXiv_src_1210_015.tar",
        "s3://arxiv/src/arXiv_src_1210_016.tar",
        "s3://arxiv/src/arXiv_src_1210_017.tar",
        "s3://arxiv/src/arXiv_src_1210_018.tar",
        "s3://arxiv/src/arXiv_src_1210_019.tar",
        "s3://arxiv/src/arXiv_src_1211_001.tar",
        "s3://arxiv/src/arXiv_src_1211_002.tar",
        "s3://arxiv/src/arXiv_src_1211_003.tar",
        "s3://arxiv/src/arXiv_src_1211_004.tar",
        "s3://arxiv/src/arXiv_src_1211_005.tar",
        "s3://arxiv/src/arXiv_src_1211_006.tar",
        "s3://arxiv/src/arXiv_src_1211_007.tar",
        "s3://arxiv/src/arXiv_src_1211_008.tar",
        "s3://arxiv/src/arXiv_src_1211_009.tar",
        "s3://arxiv/src/arXiv_src_1211_010.tar",
        "s3://arxiv/src/arXiv_src_1211_011.tar",
        "s3://arxiv/src/arXiv_src_1212_001.tar",
        "s3://arxiv/src/arXiv_src_1212_002.tar",
        "s3://arxiv/src/arXiv_src_1212_003.tar",
        "s3://arxiv/src/arXiv_src_1212_004.tar",
        "s3://arxiv/src/arXiv_src_1212_005.tar",
        "s3://arxiv/src/arXiv_src_1212_006.tar",
        "s3://arxiv/src/arXiv_src_1212_007.tar",
        "s3://arxiv/src/arXiv_src_1212_008.tar",
        "s3://arxiv/src/arXiv_src_1212_009.tar",
        "s3://arxiv/src/arXiv_src_1212_010.tar",
        "s3://arxiv/src/arXiv_src_1301_001.tar",
        "s3://arxiv/src/arXiv_src_1301_002.tar",
        "s3://arxiv/src/arXiv_src_1301_003.tar",
        "s3://arxiv/src/arXiv_src_1301_004.tar",
        "s3://arxiv/src/arXiv_src_1301_005.tar",
        "s3://arxiv/src/arXiv_src_1301_006.tar",
        "s3://arxiv/src/arXiv_src_1301_007.tar",
        "s3://arxiv/src/arXiv_src_1301_008.tar",
        "s3://arxiv/src/arXiv_src_1301_009.tar",
        "s3://arxiv/src/arXiv_src_1301_010.tar",
        "s3://arxiv/src/arXiv_src_1301_011.tar",
        "s3://arxiv/src/arXiv_src_1302_001.tar",
        "s3://arxiv/src/arXiv_src_1302_002.tar",
        "s3://arxiv/src/arXiv_src_1302_003.tar",
        "s3://arxiv/src/arXiv_src_1302_004.tar",
        "s3://arxiv/src/arXiv_src_1302_005.tar",
        "s3://arxiv/src/arXiv_src_1302_006.tar",
        "s3://arxiv/src/arXiv_src_1302_007.tar",
        "s3://arxiv/src/arXiv_src_1302_008.tar",
        "s3://arxiv/src/arXiv_src_1302_009.tar",
        "s3://arxiv/src/arXiv_src_1302_010.tar",
        "s3://arxiv/src/arXiv_src_1302_011.tar",
        "s3://arxiv/src/arXiv_src_1303_001.tar",
        "s3://arxiv/src/arXiv_src_1303_002.tar",
        "s3://arxiv/src/arXiv_src_1303_003.tar",
        "s3://arxiv/src/arXiv_src_1303_004.tar",
        "s3://arxiv/src/arXiv_src_1303_005.tar",
        "s3://arxiv/src/arXiv_src_1303_006.tar",
        "s3://arxiv/src/arXiv_src_1303_007.tar",
        "s3://arxiv/src/arXiv_src_1303_008.tar",
        "s3://arxiv/src/arXiv_src_1303_009.tar",
        "s3://arxiv/src/arXiv_src_1303_010.tar",
        "s3://arxiv/src/arXiv_src_1303_011.tar",
        "s3://arxiv/src/arXiv_src_1303_012.tar",
        "s3://arxiv/src/arXiv_src_1304_001.tar",
        "s3://arxiv/src/arXiv_src_1304_002.tar",
        "s3://arxiv/src/arXiv_src_1304_003.tar",
        "s3://arxiv/src/arXiv_src_1304_004.tar",
        "s3://arxiv/src/arXiv_src_1304_005.tar",
        "s3://arxiv/src/arXiv_src_1304_006.tar",
        "s3://arxiv/src/arXiv_src_1304_007.tar",
        "s3://arxiv/src/arXiv_src_1304_008.tar",
        "s3://arxiv/src/arXiv_src_1304_009.tar",
        "s3://arxiv/src/arXiv_src_1304_010.tar",
        "s3://arxiv/src/arXiv_src_1304_011.tar",
        "s3://arxiv/src/arXiv_src_1304_012.tar",
        "s3://arxiv/src/arXiv_src_1305_001.tar",
        "s3://arxiv/src/arXiv_src_1305_002.tar",
        "s3://arxiv/src/arXiv_src_1305_003.tar",
        "s3://arxiv/src/arXiv_src_1305_004.tar",
        "s3://arxiv/src/arXiv_src_1305_005.tar",
        "s3://arxiv/src/arXiv_src_1305_006.tar",
        "s3://arxiv/src/arXiv_src_1305_007.tar",
        "s3://arxiv/src/arXiv_src_1305_008.tar",
        "s3://arxiv/src/arXiv_src_1305_009.tar",
        "s3://arxiv/src/arXiv_src_1305_010.tar",
        "s3://arxiv/src/arXiv_src_1305_011.tar",
        "s3://arxiv/src/arXiv_src_1305_012.tar",
        "s3://arxiv/src/arXiv_src_1306_001.tar",
        "s3://arxiv/src/arXiv_src_1306_002.tar",
        "s3://arxiv/src/arXiv_src_1306_003.tar",
        "s3://arxiv/src/arXiv_src_1306_004.tar",
        "s3://arxiv/src/arXiv_src_1306_005.tar",
        "s3://arxiv/src/arXiv_src_1306_006.tar",
        "s3://arxiv/src/arXiv_src_1306_007.tar",
        "s3://arxiv/src/arXiv_src_1306_008.tar",
        "s3://arxiv/src/arXiv_src_1306_009.tar",
        "s3://arxiv/src/arXiv_src_1306_010.tar",
        "s3://arxiv/src/arXiv_src_1306_011.tar",
        "s3://arxiv/src/arXiv_src_1306_012.tar",
        "s3://arxiv/src/arXiv_src_1307_001.tar",
        "s3://arxiv/src/arXiv_src_1307_002.tar",
        "s3://arxiv/src/arXiv_src_1307_003.tar",
        "s3://arxiv/src/arXiv_src_1307_004.tar",
        "s3://arxiv/src/arXiv_src_1307_005.tar",
        "s3://arxiv/src/arXiv_src_1307_006.tar",
        "s3://arxiv/src/arXiv_src_1307_007.tar",
        "s3://arxiv/src/arXiv_src_1307_008.tar",
        "s3://arxiv/src/arXiv_src_1307_009.tar",
        "s3://arxiv/src/arXiv_src_1307_010.tar",
        "s3://arxiv/src/arXiv_src_1307_011.tar",
        "s3://arxiv/src/arXiv_src_1307_012.tar",
        "s3://arxiv/src/arXiv_src_1307_013.tar",
        "s3://arxiv/src/arXiv_src_1307_014.tar",
        "s3://arxiv/src/arXiv_src_1307_015.tar",
        "s3://arxiv/src/arXiv_src_1308_001.tar",
        "s3://arxiv/src/arXiv_src_1308_002.tar",
        "s3://arxiv/src/arXiv_src_1308_003.tar",
        "s3://arxiv/src/arXiv_src_1308_004.tar",
        "s3://arxiv/src/arXiv_src_1308_005.tar",
        "s3://arxiv/src/arXiv_src_1308_006.tar",
        "s3://arxiv/src/arXiv_src_1308_007.tar",
        "s3://arxiv/src/arXiv_src_1308_008.tar",
        "s3://arxiv/src/arXiv_src_1308_009.tar",
        "s3://arxiv/src/arXiv_src_1308_010.tar",
        "s3://arxiv/src/arXiv_src_1308_011.tar",
        "s3://arxiv/src/arXiv_src_1308_012.tar",
        "s3://arxiv/src/arXiv_src_1309_001.tar",
        "s3://arxiv/src/arXiv_src_1309_002.tar",
        "s3://arxiv/src/arXiv_src_1309_003.tar",
        "s3://arxiv/src/arXiv_src_1309_004.tar",
        "s3://arxiv/src/arXiv_src_1309_005.tar",
        "s3://arxiv/src/arXiv_src_1309_006.tar",
        "s3://arxiv/src/arXiv_src_1309_007.tar",
        "s3://arxiv/src/arXiv_src_1309_008.tar",
        "s3://arxiv/src/arXiv_src_1309_009.tar",
        "s3://arxiv/src/arXiv_src_1309_010.tar",
        "s3://arxiv/src/arXiv_src_1309_011.tar",
        "s3://arxiv/src/arXiv_src_1309_012.tar",
        "s3://arxiv/src/arXiv_src_1309_013.tar",
        "s3://arxiv/src/arXiv_src_1310_001.tar",
        "s3://arxiv/src/arXiv_src_1310_002.tar",
        "s3://arxiv/src/arXiv_src_1310_003.tar",
        "s3://arxiv/src/arXiv_src_1310_004.tar",
        "s3://arxiv/src/arXiv_src_1310_005.tar",
        "s3://arxiv/src/arXiv_src_1310_006.tar",
        "s3://arxiv/src/arXiv_src_1310_007.tar",
        "s3://arxiv/src/arXiv_src_1310_008.tar",
        "s3://arxiv/src/arXiv_src_1310_009.tar",
        "s3://arxiv/src/arXiv_src_1310_010.tar",
        "s3://arxiv/src/arXiv_src_1310_011.tar",
        "s3://arxiv/src/arXiv_src_1310_012.tar",
        "s3://arxiv/src/arXiv_src_1310_013.tar",
        "s3://arxiv/src/arXiv_src_1310_014.tar",
        "s3://arxiv/src/arXiv_src_1310_015.tar",
        "s3://arxiv/src/arXiv_src_1310_016.tar",
        "s3://arxiv/src/arXiv_src_1310_017.tar",
        "s3://arxiv/src/arXiv_src_1310_018.tar",
        "s3://arxiv/src/arXiv_src_1310_019.tar",
        "s3://arxiv/src/arXiv_src_1310_020.tar",
        "s3://arxiv/src/arXiv_src_1310_021.tar",
        "s3://arxiv/src/arXiv_src_1311_001.tar",
        "s3://arxiv/src/arXiv_src_1311_002.tar",
        "s3://arxiv/src/arXiv_src_1311_003.tar",
        "s3://arxiv/src/arXiv_src_1311_004.tar",
        "s3://arxiv/src/arXiv_src_1311_005.tar",
        "s3://arxiv/src/arXiv_src_1311_006.tar",
        "s3://arxiv/src/arXiv_src_1311_007.tar",
        "s3://arxiv/src/arXiv_src_1311_008.tar",
        "s3://arxiv/src/arXiv_src_1311_009.tar",
        "s3://arxiv/src/arXiv_src_1311_010.tar",
        "s3://arxiv/src/arXiv_src_1311_011.tar",
        "s3://arxiv/src/arXiv_src_1311_012.tar",
        "s3://arxiv/src/arXiv_src_1311_013.tar",
        "s3://arxiv/src/arXiv_src_1312_001.tar",
        "s3://arxiv/src/arXiv_src_1312_002.tar",
        "s3://arxiv/src/arXiv_src_1312_003.tar",
        "s3://arxiv/src/arXiv_src_1312_004.tar",
        "s3://arxiv/src/arXiv_src_1312_005.tar",
        "s3://arxiv/src/arXiv_src_1312_006.tar",
        "s3://arxiv/src/arXiv_src_1312_007.tar",
        "s3://arxiv/src/arXiv_src_1312_008.tar",
        "s3://arxiv/src/arXiv_src_1312_009.tar",
        "s3://arxiv/src/arXiv_src_1312_010.tar",
        "s3://arxiv/src/arXiv_src_1312_011.tar",
        "s3://arxiv/src/arXiv_src_1312_012.tar",
        "s3://arxiv/src/arXiv_src_1401_001.tar",
        "s3://arxiv/src/arXiv_src_1401_002.tar",
        "s3://arxiv/src/arXiv_src_1401_003.tar",
        "s3://arxiv/src/arXiv_src_1401_004.tar",
        "s3://arxiv/src/arXiv_src_1401_005.tar",
        "s3://arxiv/src/arXiv_src_1401_006.tar",
        "s3://arxiv/src/arXiv_src_1401_007.tar",
        "s3://arxiv/src/arXiv_src_1401_008.tar",
        "s3://arxiv/src/arXiv_src_1401_009.tar",
        "s3://arxiv/src/arXiv_src_1401_010.tar",
        "s3://arxiv/src/arXiv_src_1401_011.tar",
        "s3://arxiv/src/arXiv_src_1401_012.tar",
        "s3://arxiv/src/arXiv_src_1401_013.tar",
        "s3://arxiv/src/arXiv_src_1402_001.tar",
        "s3://arxiv/src/arXiv_src_1402_002.tar",
        "s3://arxiv/src/arXiv_src_1402_003.tar",
        "s3://arxiv/src/arXiv_src_1402_004.tar",
        "s3://arxiv/src/arXiv_src_1402_005.tar",
        "s3://arxiv/src/arXiv_src_1402_006.tar",
        "s3://arxiv/src/arXiv_src_1402_007.tar",
        "s3://arxiv/src/arXiv_src_1402_008.tar",
        "s3://arxiv/src/arXiv_src_1402_009.tar",
        "s3://arxiv/src/arXiv_src_1402_010.tar",
        "s3://arxiv/src/arXiv_src_1402_011.tar",
        "s3://arxiv/src/arXiv_src_1402_012.tar",
        "s3://arxiv/src/arXiv_src_1402_013.tar",
        "s3://arxiv/src/arXiv_src_1403_001.tar",
        "s3://arxiv/src/arXiv_src_1403_002.tar",
        "s3://arxiv/src/arXiv_src_1403_003.tar",
        "s3://arxiv/src/arXiv_src_1403_004.tar",
        "s3://arxiv/src/arXiv_src_1403_005.tar",
        "s3://arxiv/src/arXiv_src_1403_006.tar",
        "s3://arxiv/src/arXiv_src_1403_007.tar",
        "s3://arxiv/src/arXiv_src_1403_008.tar",
        "s3://arxiv/src/arXiv_src_1403_009.tar",
        "s3://arxiv/src/arXiv_src_1403_010.tar",
        "s3://arxiv/src/arXiv_src_1403_011.tar",
        "s3://arxiv/src/arXiv_src_1403_012.tar",
        "s3://arxiv/src/arXiv_src_1403_013.tar",
        "s3://arxiv/src/arXiv_src_1403_014.tar",
        "s3://arxiv/src/arXiv_src_1404_001.tar",
        "s3://arxiv/src/arXiv_src_1404_002.tar",
        "s3://arxiv/src/arXiv_src_1404_003.tar",
        "s3://arxiv/src/arXiv_src_1404_004.tar",
        "s3://arxiv/src/arXiv_src_1404_005.tar",
        "s3://arxiv/src/arXiv_src_1404_006.tar",
        "s3://arxiv/src/arXiv_src_1404_007.tar",
        "s3://arxiv/src/arXiv_src_1404_008.tar",
        "s3://arxiv/src/arXiv_src_1404_009.tar",
        "s3://arxiv/src/arXiv_src_1404_010.tar",
        "s3://arxiv/src/arXiv_src_1404_011.tar",
        "s3://arxiv/src/arXiv_src_1404_012.tar",
        "s3://arxiv/src/arXiv_src_1404_013.tar",
        "s3://arxiv/src/arXiv_src_1405_001.tar",
        "s3://arxiv/src/arXiv_src_1405_002.tar",
        "s3://arxiv/src/arXiv_src_1405_003.tar",
        "s3://arxiv/src/arXiv_src_1405_004.tar",
        "s3://arxiv/src/arXiv_src_1405_005.tar",
        "s3://arxiv/src/arXiv_src_1405_006.tar",
        "s3://arxiv/src/arXiv_src_1405_007.tar",
        "s3://arxiv/src/arXiv_src_1405_008.tar",
        "s3://arxiv/src/arXiv_src_1405_009.tar",
        "s3://arxiv/src/arXiv_src_1405_010.tar",
        "s3://arxiv/src/arXiv_src_1405_011.tar",
        "s3://arxiv/src/arXiv_src_1405_012.tar",
        "s3://arxiv/src/arXiv_src_1405_013.tar",
        "s3://arxiv/src/arXiv_src_1405_014.tar",
        "s3://arxiv/src/arXiv_src_1406_001.tar",
        "s3://arxiv/src/arXiv_src_1406_002.tar",
        "s3://arxiv/src/arXiv_src_1406_003.tar",
        "s3://arxiv/src/arXiv_src_1406_004.tar",
        "s3://arxiv/src/arXiv_src_1406_005.tar",
        "s3://arxiv/src/arXiv_src_1406_006.tar",
        "s3://arxiv/src/arXiv_src_1406_007.tar",
        "s3://arxiv/src/arXiv_src_1406_008.tar",
        "s3://arxiv/src/arXiv_src_1406_009.tar",
        "s3://arxiv/src/arXiv_src_1406_010.tar",
        "s3://arxiv/src/arXiv_src_1406_011.tar",
        "s3://arxiv/src/arXiv_src_1406_012.tar",
        "s3://arxiv/src/arXiv_src_1406_013.tar",
        "s3://arxiv/src/arXiv_src_1406_014.tar",
        "s3://arxiv/src/arXiv_src_1407_001.tar",
        "s3://arxiv/src/arXiv_src_1407_002.tar",
        "s3://arxiv/src/arXiv_src_1407_003.tar",
        "s3://arxiv/src/arXiv_src_1407_004.tar",
        "s3://arxiv/src/arXiv_src_1407_005.tar",
        "s3://arxiv/src/arXiv_src_1407_006.tar",
        "s3://arxiv/src/arXiv_src_1407_007.tar",
        "s3://arxiv/src/arXiv_src_1407_008.tar",
        "s3://arxiv/src/arXiv_src_1407_009.tar",
        "s3://arxiv/src/arXiv_src_1407_010.tar",
        "s3://arxiv/src/arXiv_src_1407_011.tar",
        "s3://arxiv/src/arXiv_src_1407_012.tar",
        "s3://arxiv/src/arXiv_src_1407_013.tar",
        "s3://arxiv/src/arXiv_src_1407_014.tar",
        "s3://arxiv/src/arXiv_src_1407_015.tar",
        "s3://arxiv/src/arXiv_src_1407_016.tar",
        "s3://arxiv/src/arXiv_src_1408_001.tar",
        "s3://arxiv/src/arXiv_src_1408_002.tar",
        "s3://arxiv/src/arXiv_src_1408_003.tar",
        "s3://arxiv/src/arXiv_src_1408_004.tar",
        "s3://arxiv/src/arXiv_src_1408_005.tar",
        "s3://arxiv/src/arXiv_src_1408_006.tar",
        "s3://arxiv/src/arXiv_src_1408_007.tar",
        "s3://arxiv/src/arXiv_src_1408_008.tar",
        "s3://arxiv/src/arXiv_src_1408_009.tar",
        "s3://arxiv/src/arXiv_src_1408_010.tar",
        "s3://arxiv/src/arXiv_src_1408_011.tar",
        "s3://arxiv/src/arXiv_src_1408_012.tar",
        "s3://arxiv/src/arXiv_src_1408_013.tar",
        "s3://arxiv/src/arXiv_src_1409_001.tar",
        "s3://arxiv/src/arXiv_src_1409_002.tar",
        "s3://arxiv/src/arXiv_src_1409_003.tar",
        "s3://arxiv/src/arXiv_src_1409_004.tar",
        "s3://arxiv/src/arXiv_src_1409_005.tar",
        "s3://arxiv/src/arXiv_src_1409_006.tar",
        "s3://arxiv/src/arXiv_src_1409_007.tar",
        "s3://arxiv/src/arXiv_src_1409_008.tar",
        "s3://arxiv/src/arXiv_src_1409_009.tar",
        "s3://arxiv/src/arXiv_src_1409_010.tar",
        "s3://arxiv/src/arXiv_src_1409_011.tar",
        "s3://arxiv/src/arXiv_src_1409_012.tar",
        "s3://arxiv/src/arXiv_src_1409_013.tar",
        "s3://arxiv/src/arXiv_src_1409_014.tar",
        "s3://arxiv/src/arXiv_src_1409_015.tar",
        "s3://arxiv/src/arXiv_src_1409_016.tar",
        "s3://arxiv/src/arXiv_src_1410_001.tar",
        "s3://arxiv/src/arXiv_src_1410_002.tar",
        "s3://arxiv/src/arXiv_src_1410_003.tar",
        "s3://arxiv/src/arXiv_src_1410_004.tar",
        "s3://arxiv/src/arXiv_src_1410_005.tar",
        "s3://arxiv/src/arXiv_src_1410_006.tar",
        "s3://arxiv/src/arXiv_src_1410_007.tar",
        "s3://arxiv/src/arXiv_src_1410_008.tar",
        "s3://arxiv/src/arXiv_src_1410_009.tar",
        "s3://arxiv/src/arXiv_src_1410_010.tar",
        "s3://arxiv/src/arXiv_src_1410_011.tar",
        "s3://arxiv/src/arXiv_src_1410_012.tar",
        "s3://arxiv/src/arXiv_src_1410_013.tar",
        "s3://arxiv/src/arXiv_src_1410_014.tar",
        "s3://arxiv/src/arXiv_src_1410_015.tar",
        "s3://arxiv/src/arXiv_src_1410_016.tar",
        "s3://arxiv/src/arXiv_src_1410_017.tar",
        "s3://arxiv/src/arXiv_src_1411_001.tar",
        "s3://arxiv/src/arXiv_src_1411_002.tar",
        "s3://arxiv/src/arXiv_src_1411_003.tar",
        "s3://arxiv/src/arXiv_src_1411_004.tar",
        "s3://arxiv/src/arXiv_src_1411_005.tar",
        "s3://arxiv/src/arXiv_src_1411_006.tar",
        "s3://arxiv/src/arXiv_src_1411_007.tar",
        "s3://arxiv/src/arXiv_src_1411_008.tar",
        "s3://arxiv/src/arXiv_src_1411_009.tar",
        "s3://arxiv/src/arXiv_src_1411_010.tar",
        "s3://arxiv/src/arXiv_src_1411_011.tar",
        "s3://arxiv/src/arXiv_src_1411_012.tar",
        "s3://arxiv/src/arXiv_src_1411_013.tar",
        "s3://arxiv/src/arXiv_src_1411_014.tar",
        "s3://arxiv/src/arXiv_src_1411_015.tar",
        "s3://arxiv/src/arXiv_src_1412_001.tar",
        "s3://arxiv/src/arXiv_src_1412_002.tar",
        "s3://arxiv/src/arXiv_src_1412_003.tar",
        "s3://arxiv/src/arXiv_src_1412_004.tar",
        "s3://arxiv/src/arXiv_src_1412_005.tar",
        "s3://arxiv/src/arXiv_src_1412_006.tar",
        "s3://arxiv/src/arXiv_src_1412_007.tar",
        "s3://arxiv/src/arXiv_src_1412_008.tar",
        "s3://arxiv/src/arXiv_src_1412_009.tar",
        "s3://arxiv/src/arXiv_src_1412_010.tar",
        "s3://arxiv/src/arXiv_src_1412_011.tar",
        "s3://arxiv/src/arXiv_src_1412_012.tar",
        "s3://arxiv/src/arXiv_src_1412_013.tar",
        "s3://arxiv/src/arXiv_src_1412_014.tar",
        "s3://arxiv/src/arXiv_src_1412_015.tar",
        "s3://arxiv/src/arXiv_src_1412_016.tar",
        "s3://arxiv/src/arXiv_src_1412_017.tar",
        "s3://arxiv/src/arXiv_src_1501_001.tar",
        "s3://arxiv/src/arXiv_src_1501_002.tar",
        "s3://arxiv/src/arXiv_src_1501_003.tar",
        "s3://arxiv/src/arXiv_src_1501_004.tar",
        "s3://arxiv/src/arXiv_src_1501_005.tar",
        "s3://arxiv/src/arXiv_src_1501_006.tar",
        "s3://arxiv/src/arXiv_src_1501_007.tar",
        "s3://arxiv/src/arXiv_src_1501_008.tar",
        "s3://arxiv/src/arXiv_src_1501_009.tar",
        "s3://arxiv/src/arXiv_src_1501_010.tar",
        "s3://arxiv/src/arXiv_src_1501_011.tar",
        "s3://arxiv/src/arXiv_src_1501_012.tar",
        "s3://arxiv/src/arXiv_src_1501_013.tar",
        "s3://arxiv/src/arXiv_src_1501_014.tar",
        "s3://arxiv/src/arXiv_src_1501_015.tar",
        "s3://arxiv/src/arXiv_src_1502_001.tar",
        "s3://arxiv/src/arXiv_src_1502_002.tar",
        "s3://arxiv/src/arXiv_src_1502_003.tar",
        "s3://arxiv/src/arXiv_src_1502_004.tar",
        "s3://arxiv/src/arXiv_src_1502_005.tar",
        "s3://arxiv/src/arXiv_src_1502_006.tar",
        "s3://arxiv/src/arXiv_src_1502_007.tar",
        "s3://arxiv/src/arXiv_src_1502_008.tar",
        "s3://arxiv/src/arXiv_src_1502_009.tar",
        "s3://arxiv/src/arXiv_src_1502_010.tar",
        "s3://arxiv/src/arXiv_src_1502_011.tar",
        "s3://arxiv/src/arXiv_src_1502_012.tar",
        "s3://arxiv/src/arXiv_src_1502_013.tar",
        "s3://arxiv/src/arXiv_src_1502_014.tar",
        "s3://arxiv/src/arXiv_src_1502_015.tar",
        "s3://arxiv/src/arXiv_src_1502_016.tar",
        "s3://arxiv/src/arXiv_src_1503_001.tar",
        "s3://arxiv/src/arXiv_src_1503_002.tar",
        "s3://arxiv/src/arXiv_src_1503_003.tar",
        "s3://arxiv/src/arXiv_src_1503_004.tar",
        "s3://arxiv/src/arXiv_src_1503_005.tar",
        "s3://arxiv/src/arXiv_src_1503_006.tar",
        "s3://arxiv/src/arXiv_src_1503_007.tar",
        "s3://arxiv/src/arXiv_src_1503_008.tar",
        "s3://arxiv/src/arXiv_src_1503_009.tar",
        "s3://arxiv/src/arXiv_src_1503_010.tar",
        "s3://arxiv/src/arXiv_src_1503_011.tar",
        "s3://arxiv/src/arXiv_src_1503_012.tar",
        "s3://arxiv/src/arXiv_src_1503_013.tar",
        "s3://arxiv/src/arXiv_src_1503_014.tar",
        "s3://arxiv/src/arXiv_src_1503_015.tar",
        "s3://arxiv/src/arXiv_src_1503_016.tar",
        "s3://arxiv/src/arXiv_src_1503_017.tar",
        "s3://arxiv/src/arXiv_src_1503_018.tar",
        "s3://arxiv/src/arXiv_src_1504_001.tar",
        "s3://arxiv/src/arXiv_src_1504_002.tar",
        "s3://arxiv/src/arXiv_src_1504_003.tar",
        "s3://arxiv/src/arXiv_src_1504_004.tar",
        "s3://arxiv/src/arXiv_src_1504_005.tar",
        "s3://arxiv/src/arXiv_src_1504_006.tar",
        "s3://arxiv/src/arXiv_src_1504_007.tar",
        "s3://arxiv/src/arXiv_src_1504_008.tar",
        "s3://arxiv/src/arXiv_src_1504_009.tar",
        "s3://arxiv/src/arXiv_src_1504_010.tar",
        "s3://arxiv/src/arXiv_src_1504_011.tar",
        "s3://arxiv/src/arXiv_src_1504_012.tar",
        "s3://arxiv/src/arXiv_src_1504_013.tar",
        "s3://arxiv/src/arXiv_src_1504_014.tar",
        "s3://arxiv/src/arXiv_src_1504_015.tar",
        "s3://arxiv/src/arXiv_src_1504_016.tar",
        "s3://arxiv/src/arXiv_src_1504_017.tar",
        "s3://arxiv/src/arXiv_src_1505_001.tar",
        "s3://arxiv/src/arXiv_src_1505_002.tar",
        "s3://arxiv/src/arXiv_src_1505_003.tar",
        "s3://arxiv/src/arXiv_src_1505_004.tar",
        "s3://arxiv/src/arXiv_src_1505_005.tar",
        "s3://arxiv/src/arXiv_src_1505_006.tar",
        "s3://arxiv/src/arXiv_src_1505_007.tar",
        "s3://arxiv/src/arXiv_src_1505_008.tar",
        "s3://arxiv/src/arXiv_src_1505_009.tar",
        "s3://arxiv/src/arXiv_src_1505_010.tar",
        "s3://arxiv/src/arXiv_src_1505_011.tar",
        "s3://arxiv/src/arXiv_src_1505_012.tar",
        "s3://arxiv/src/arXiv_src_1505_013.tar",
        "s3://arxiv/src/arXiv_src_1505_014.tar",
        "s3://arxiv/src/arXiv_src_1505_015.tar",
        "s3://arxiv/src/arXiv_src_1505_016.tar",
        "s3://arxiv/src/arXiv_src_1505_017.tar",
        "s3://arxiv/src/arXiv_src_1506_001.tar",
        "s3://arxiv/src/arXiv_src_1506_002.tar",
        "s3://arxiv/src/arXiv_src_1506_003.tar",
        "s3://arxiv/src/arXiv_src_1506_004.tar",
        "s3://arxiv/src/arXiv_src_1506_005.tar",
        "s3://arxiv/src/arXiv_src_1506_006.tar",
        "s3://arxiv/src/arXiv_src_1506_007.tar",
        "s3://arxiv/src/arXiv_src_1506_008.tar",
        "s3://arxiv/src/arXiv_src_1506_009.tar",
        "s3://arxiv/src/arXiv_src_1506_010.tar",
        "s3://arxiv/src/arXiv_src_1506_011.tar",
        "s3://arxiv/src/arXiv_src_1506_012.tar",
        "s3://arxiv/src/arXiv_src_1506_013.tar",
        "s3://arxiv/src/arXiv_src_1506_014.tar",
        "s3://arxiv/src/arXiv_src_1506_015.tar",
        "s3://arxiv/src/arXiv_src_1506_016.tar",
        "s3://arxiv/src/arXiv_src_1506_017.tar",
        "s3://arxiv/src/arXiv_src_1506_018.tar",
        "s3://arxiv/src/arXiv_src_1507_001.tar",
        "s3://arxiv/src/arXiv_src_1507_002.tar",
        "s3://arxiv/src/arXiv_src_1507_003.tar",
        "s3://arxiv/src/arXiv_src_1507_004.tar",
        "s3://arxiv/src/arXiv_src_1507_005.tar",
        "s3://arxiv/src/arXiv_src_1507_006.tar",
        "s3://arxiv/src/arXiv_src_1507_007.tar",
        "s3://arxiv/src/arXiv_src_1507_008.tar",
        "s3://arxiv/src/arXiv_src_1507_009.tar",
        "s3://arxiv/src/arXiv_src_1507_010.tar",
        "s3://arxiv/src/arXiv_src_1507_011.tar",
        "s3://arxiv/src/arXiv_src_1507_012.tar",
        "s3://arxiv/src/arXiv_src_1507_013.tar",
        "s3://arxiv/src/arXiv_src_1507_014.tar",
        "s3://arxiv/src/arXiv_src_1507_015.tar",
        "s3://arxiv/src/arXiv_src_1507_016.tar",
        "s3://arxiv/src/arXiv_src_1507_017.tar",
        "s3://arxiv/src/arXiv_src_1507_018.tar",
        "s3://arxiv/src/arXiv_src_1508_001.tar",
        "s3://arxiv/src/arXiv_src_1508_002.tar",
        "s3://arxiv/src/arXiv_src_1508_003.tar",
        "s3://arxiv/src/arXiv_src_1508_004.tar",
        "s3://arxiv/src/arXiv_src_1508_005.tar",
        "s3://arxiv/src/arXiv_src_1508_006.tar",
        "s3://arxiv/src/arXiv_src_1508_007.tar",
        "s3://arxiv/src/arXiv_src_1508_008.tar",
        "s3://arxiv/src/arXiv_src_1508_009.tar",
        "s3://arxiv/src/arXiv_src_1508_010.tar",
        "s3://arxiv/src/arXiv_src_1508_011.tar",
        "s3://arxiv/src/arXiv_src_1508_012.tar",
        "s3://arxiv/src/arXiv_src_1508_013.tar",
        "s3://arxiv/src/arXiv_src_1508_014.tar",
        "s3://arxiv/src/arXiv_src_1508_015.tar",
        "s3://arxiv/src/arXiv_src_1508_016.tar",
        "s3://arxiv/src/arXiv_src_1509_001.tar",
        "s3://arxiv/src/arXiv_src_1509_002.tar",
        "s3://arxiv/src/arXiv_src_1509_003.tar",
        "s3://arxiv/src/arXiv_src_1509_004.tar",
        "s3://arxiv/src/arXiv_src_1509_005.tar",
        "s3://arxiv/src/arXiv_src_1509_006.tar",
        "s3://arxiv/src/arXiv_src_1509_007.tar",
        "s3://arxiv/src/arXiv_src_1509_008.tar",
        "s3://arxiv/src/arXiv_src_1509_009.tar",
        "s3://arxiv/src/arXiv_src_1509_010.tar",
        "s3://arxiv/src/arXiv_src_1509_011.tar",
        "s3://arxiv/src/arXiv_src_1509_012.tar",
        "s3://arxiv/src/arXiv_src_1509_013.tar",
        "s3://arxiv/src/arXiv_src_1509_014.tar",
        "s3://arxiv/src/arXiv_src_1509_015.tar",
        "s3://arxiv/src/arXiv_src_1509_016.tar",
        "s3://arxiv/src/arXiv_src_1509_017.tar",
        "s3://arxiv/src/arXiv_src_1509_018.tar",
        "s3://arxiv/src/arXiv_src_1509_019.tar",
        "s3://arxiv/src/arXiv_src_1509_020.tar",
        "s3://arxiv/src/arXiv_src_1510_001.tar",
        "s3://arxiv/src/arXiv_src_1510_002.tar",
        "s3://arxiv/src/arXiv_src_1510_003.tar",
        "s3://arxiv/src/arXiv_src_1510_004.tar",
        "s3://arxiv/src/arXiv_src_1510_005.tar",
        "s3://arxiv/src/arXiv_src_1510_006.tar",
        "s3://arxiv/src/arXiv_src_1510_007.tar",
        "s3://arxiv/src/arXiv_src_1510_008.tar",
        "s3://arxiv/src/arXiv_src_1510_009.tar",
        "s3://arxiv/src/arXiv_src_1510_010.tar",
        "s3://arxiv/src/arXiv_src_1510_011.tar",
        "s3://arxiv/src/arXiv_src_1510_012.tar",
        "s3://arxiv/src/arXiv_src_1510_013.tar",
        "s3://arxiv/src/arXiv_src_1510_014.tar",
        "s3://arxiv/src/arXiv_src_1510_015.tar",
        "s3://arxiv/src/arXiv_src_1510_016.tar",
        "s3://arxiv/src/arXiv_src_1510_017.tar",
        "s3://arxiv/src/arXiv_src_1510_018.tar",
        "s3://arxiv/src/arXiv_src_1510_019.tar",
        "s3://arxiv/src/arXiv_src_1511_001.tar",
        "s3://arxiv/src/arXiv_src_1511_002.tar",
        "s3://arxiv/src/arXiv_src_1511_003.tar",
        "s3://arxiv/src/arXiv_src_1511_004.tar",
        "s3://arxiv/src/arXiv_src_1511_005.tar",
        "s3://arxiv/src/arXiv_src_1511_006.tar",
        "s3://arxiv/src/arXiv_src_1511_007.tar",
        "s3://arxiv/src/arXiv_src_1511_008.tar",
        "s3://arxiv/src/arXiv_src_1511_009.tar",
        "s3://arxiv/src/arXiv_src_1511_010.tar",
        "s3://arxiv/src/arXiv_src_1511_011.tar",
        "s3://arxiv/src/arXiv_src_1511_012.tar",
        "s3://arxiv/src/arXiv_src_1511_013.tar",
        "s3://arxiv/src/arXiv_src_1511_014.tar",
        "s3://arxiv/src/arXiv_src_1511_015.tar",
        "s3://arxiv/src/arXiv_src_1511_016.tar",
        "s3://arxiv/src/arXiv_src_1511_017.tar",
        "s3://arxiv/src/arXiv_src_1511_018.tar",
        "s3://arxiv/src/arXiv_src_1511_019.tar",
        "s3://arxiv/src/arXiv_src_1511_020.tar",
        "s3://arxiv/src/arXiv_src_1511_021.tar",
        "s3://arxiv/src/arXiv_src_1512_001.tar",
        "s3://arxiv/src/arXiv_src_1512_002.tar",
        "s3://arxiv/src/arXiv_src_1512_003.tar",
        "s3://arxiv/src/arXiv_src_1512_004.tar",
        "s3://arxiv/src/arXiv_src_1512_005.tar",
        "s3://arxiv/src/arXiv_src_1512_006.tar",
        "s3://arxiv/src/arXiv_src_1512_007.tar",
        "s3://arxiv/src/arXiv_src_1512_008.tar",
        "s3://arxiv/src/arXiv_src_1512_009.tar",
        "s3://arxiv/src/arXiv_src_1512_010.tar",
        "s3://arxiv/src/arXiv_src_1512_011.tar",
        "s3://arxiv/src/arXiv_src_1512_012.tar",
        "s3://arxiv/src/arXiv_src_1512_013.tar",
        "s3://arxiv/src/arXiv_src_1512_014.tar",
        "s3://arxiv/src/arXiv_src_1512_015.tar",
        "s3://arxiv/src/arXiv_src_1512_016.tar",
        "s3://arxiv/src/arXiv_src_1512_017.tar",
        "s3://arxiv/src/arXiv_src_1512_018.tar",
        "s3://arxiv/src/arXiv_src_1512_019.tar",
        "s3://arxiv/src/arXiv_src_1601_001.tar",
        "s3://arxiv/src/arXiv_src_1601_002.tar",
        "s3://arxiv/src/arXiv_src_1601_003.tar",
        "s3://arxiv/src/arXiv_src_1601_004.tar",
        "s3://arxiv/src/arXiv_src_1601_005.tar",
        "s3://arxiv/src/arXiv_src_1601_006.tar",
        "s3://arxiv/src/arXiv_src_1601_007.tar",
        "s3://arxiv/src/arXiv_src_1601_008.tar",
        "s3://arxiv/src/arXiv_src_1601_009.tar",
        "s3://arxiv/src/arXiv_src_1601_010.tar",
        "s3://arxiv/src/arXiv_src_1601_011.tar",
        "s3://arxiv/src/arXiv_src_1601_012.tar",
        "s3://arxiv/src/arXiv_src_1601_013.tar",
        "s3://arxiv/src/arXiv_src_1601_014.tar",
        "s3://arxiv/src/arXiv_src_1601_015.tar",
        "s3://arxiv/src/arXiv_src_1601_016.tar",
        "s3://arxiv/src/arXiv_src_1601_017.tar",
        "s3://arxiv/src/arXiv_src_1602_001.tar",
        "s3://arxiv/src/arXiv_src_1602_002.tar",
        "s3://arxiv/src/arXiv_src_1602_003.tar",
        "s3://arxiv/src/arXiv_src_1602_004.tar",
        "s3://arxiv/src/arXiv_src_1602_005.tar",
        "s3://arxiv/src/arXiv_src_1602_006.tar",
        "s3://arxiv/src/arXiv_src_1602_007.tar",
        "s3://arxiv/src/arXiv_src_1602_008.tar",
        "s3://arxiv/src/arXiv_src_1602_009.tar",
        "s3://arxiv/src/arXiv_src_1602_010.tar",
        "s3://arxiv/src/arXiv_src_1602_011.tar",
        "s3://arxiv/src/arXiv_src_1602_012.tar",
        "s3://arxiv/src/arXiv_src_1602_013.tar",
        "s3://arxiv/src/arXiv_src_1602_014.tar",
        "s3://arxiv/src/arXiv_src_1602_015.tar",
        "s3://arxiv/src/arXiv_src_1602_016.tar",
        "s3://arxiv/src/arXiv_src_1602_017.tar",
        "s3://arxiv/src/arXiv_src_1602_018.tar",
        "s3://arxiv/src/arXiv_src_1602_019.tar",
        "s3://arxiv/src/arXiv_src_1603_001.tar",
        "s3://arxiv/src/arXiv_src_1603_002.tar",
        "s3://arxiv/src/arXiv_src_1603_003.tar",
        "s3://arxiv/src/arXiv_src_1603_004.tar",
        "s3://arxiv/src/arXiv_src_1603_005.tar",
        "s3://arxiv/src/arXiv_src_1603_006.tar",
        "s3://arxiv/src/arXiv_src_1603_007.tar",
        "s3://arxiv/src/arXiv_src_1603_008.tar",
        "s3://arxiv/src/arXiv_src_1603_009.tar",
        "s3://arxiv/src/arXiv_src_1603_010.tar",
        "s3://arxiv/src/arXiv_src_1603_011.tar",
        "s3://arxiv/src/arXiv_src_1603_012.tar",
        "s3://arxiv/src/arXiv_src_1603_013.tar",
        "s3://arxiv/src/arXiv_src_1603_014.tar",
        "s3://arxiv/src/arXiv_src_1603_015.tar",
        "s3://arxiv/src/arXiv_src_1603_016.tar",
        "s3://arxiv/src/arXiv_src_1603_017.tar",
        "s3://arxiv/src/arXiv_src_1603_018.tar",
        "s3://arxiv/src/arXiv_src_1603_019.tar",
        "s3://arxiv/src/arXiv_src_1603_020.tar",
        "s3://arxiv/src/arXiv_src_1603_021.tar",
        "s3://arxiv/src/arXiv_src_1604_001.tar",
        "s3://arxiv/src/arXiv_src_1604_002.tar",
        "s3://arxiv/src/arXiv_src_1604_003.tar",
        "s3://arxiv/src/arXiv_src_1604_004.tar",
        "s3://arxiv/src/arXiv_src_1604_005.tar",
        "s3://arxiv/src/arXiv_src_1604_006.tar",
        "s3://arxiv/src/arXiv_src_1604_007.tar",
        "s3://arxiv/src/arXiv_src_1604_008.tar",
        "s3://arxiv/src/arXiv_src_1604_009.tar",
        "s3://arxiv/src/arXiv_src_1604_010.tar",
        "s3://arxiv/src/arXiv_src_1604_011.tar",
        "s3://arxiv/src/arXiv_src_1604_012.tar",
        "s3://arxiv/src/arXiv_src_1604_013.tar",
        "s3://arxiv/src/arXiv_src_1604_014.tar",
        "s3://arxiv/src/arXiv_src_1604_015.tar",
        "s3://arxiv/src/arXiv_src_1604_016.tar",
        "s3://arxiv/src/arXiv_src_1604_017.tar",
        "s3://arxiv/src/arXiv_src_1604_018.tar",
        "s3://arxiv/src/arXiv_src_1604_019.tar",
        "s3://arxiv/src/arXiv_src_1604_020.tar",
        "s3://arxiv/src/arXiv_src_1604_021.tar",
        "s3://arxiv/src/arXiv_src_1605_001.tar",
        "s3://arxiv/src/arXiv_src_1605_002.tar",
        "s3://arxiv/src/arXiv_src_1605_003.tar",
        "s3://arxiv/src/arXiv_src_1605_004.tar",
        "s3://arxiv/src/arXiv_src_1605_005.tar",
        "s3://arxiv/src/arXiv_src_1605_006.tar",
        "s3://arxiv/src/arXiv_src_1605_007.tar",
        "s3://arxiv/src/arXiv_src_1605_008.tar",
        "s3://arxiv/src/arXiv_src_1605_009.tar",
        "s3://arxiv/src/arXiv_src_1605_010.tar",
        "s3://arxiv/src/arXiv_src_1605_011.tar",
        "s3://arxiv/src/arXiv_src_1605_012.tar",
        "s3://arxiv/src/arXiv_src_1605_013.tar",
        "s3://arxiv/src/arXiv_src_1605_014.tar",
        "s3://arxiv/src/arXiv_src_1605_015.tar",
        "s3://arxiv/src/arXiv_src_1605_016.tar",
        "s3://arxiv/src/arXiv_src_1605_017.tar",
        "s3://arxiv/src/arXiv_src_1605_018.tar",
        "s3://arxiv/src/arXiv_src_1605_019.tar",
        "s3://arxiv/src/arXiv_src_1605_020.tar",
        "s3://arxiv/src/arXiv_src_1605_021.tar",
        "s3://arxiv/src/arXiv_src_1605_022.tar",
        "s3://arxiv/src/arXiv_src_1605_023.tar",
        "s3://arxiv/src/arXiv_src_1606_001.tar",
        "s3://arxiv/src/arXiv_src_1606_002.tar",
        "s3://arxiv/src/arXiv_src_1606_003.tar",
        "s3://arxiv/src/arXiv_src_1606_004.tar",
        "s3://arxiv/src/arXiv_src_1606_005.tar",
        "s3://arxiv/src/arXiv_src_1606_006.tar",
        "s3://arxiv/src/arXiv_src_1606_007.tar",
        "s3://arxiv/src/arXiv_src_1606_008.tar",
        "s3://arxiv/src/arXiv_src_1606_009.tar",
        "s3://arxiv/src/arXiv_src_1606_010.tar",
        "s3://arxiv/src/arXiv_src_1606_011.tar",
        "s3://arxiv/src/arXiv_src_1606_012.tar",
        "s3://arxiv/src/arXiv_src_1606_013.tar",
        "s3://arxiv/src/arXiv_src_1606_014.tar",
        "s3://arxiv/src/arXiv_src_1606_015.tar",
        "s3://arxiv/src/arXiv_src_1606_016.tar",
        "s3://arxiv/src/arXiv_src_1606_017.tar",
        "s3://arxiv/src/arXiv_src_1606_018.tar",
        "s3://arxiv/src/arXiv_src_1606_019.tar",
        "s3://arxiv/src/arXiv_src_1606_020.tar",
        "s3://arxiv/src/arXiv_src_1606_021.tar",
        "s3://arxiv/src/arXiv_src_1606_022.tar",
        "s3://arxiv/src/arXiv_src_1607_001.tar",
        "s3://arxiv/src/arXiv_src_1607_002.tar",
        "s3://arxiv/src/arXiv_src_1607_003.tar",
        "s3://arxiv/src/arXiv_src_1607_004.tar",
        "s3://arxiv/src/arXiv_src_1607_005.tar",
        "s3://arxiv/src/arXiv_src_1607_006.tar",
        "s3://arxiv/src/arXiv_src_1607_007.tar",
        "s3://arxiv/src/arXiv_src_1607_008.tar",
        "s3://arxiv/src/arXiv_src_1607_009.tar",
        "s3://arxiv/src/arXiv_src_1607_010.tar",
        "s3://arxiv/src/arXiv_src_1607_011.tar",
        "s3://arxiv/src/arXiv_src_1607_012.tar",
        "s3://arxiv/src/arXiv_src_1607_013.tar",
        "s3://arxiv/src/arXiv_src_1607_014.tar",
        "s3://arxiv/src/arXiv_src_1607_015.tar",
        "s3://arxiv/src/arXiv_src_1607_016.tar",
        "s3://arxiv/src/arXiv_src_1607_017.tar",
        "s3://arxiv/src/arXiv_src_1607_018.tar",
        "s3://arxiv/src/arXiv_src_1607_019.tar",
        "s3://arxiv/src/arXiv_src_1607_020.tar",
        "s3://arxiv/src/arXiv_src_1608_001.tar",
        "s3://arxiv/src/arXiv_src_1608_002.tar",
        "s3://arxiv/src/arXiv_src_1608_003.tar",
        "s3://arxiv/src/arXiv_src_1608_004.tar",
        "s3://arxiv/src/arXiv_src_1608_005.tar",
        "s3://arxiv/src/arXiv_src_1608_006.tar",
        "s3://arxiv/src/arXiv_src_1608_007.tar",
        "s3://arxiv/src/arXiv_src_1608_008.tar",
        "s3://arxiv/src/arXiv_src_1608_009.tar",
        "s3://arxiv/src/arXiv_src_1608_010.tar",
        "s3://arxiv/src/arXiv_src_1608_011.tar",
        "s3://arxiv/src/arXiv_src_1608_012.tar",
        "s3://arxiv/src/arXiv_src_1608_013.tar",
        "s3://arxiv/src/arXiv_src_1608_014.tar",
        "s3://arxiv/src/arXiv_src_1608_015.tar",
        "s3://arxiv/src/arXiv_src_1608_016.tar",
        "s3://arxiv/src/arXiv_src_1608_017.tar",
        "s3://arxiv/src/arXiv_src_1608_018.tar",
        "s3://arxiv/src/arXiv_src_1608_019.tar",
        "s3://arxiv/src/arXiv_src_1608_020.tar",
        "s3://arxiv/src/arXiv_src_1608_021.tar",
        "s3://arxiv/src/arXiv_src_1609_001.tar",
        "s3://arxiv/src/arXiv_src_1609_002.tar",
        "s3://arxiv/src/arXiv_src_1609_003.tar",
        "s3://arxiv/src/arXiv_src_1609_004.tar",
        "s3://arxiv/src/arXiv_src_1609_005.tar",
        "s3://arxiv/src/arXiv_src_1609_006.tar",
        "s3://arxiv/src/arXiv_src_1609_007.tar",
        "s3://arxiv/src/arXiv_src_1609_008.tar",
        "s3://arxiv/src/arXiv_src_1609_009.tar",
        "s3://arxiv/src/arXiv_src_1609_010.tar",
        "s3://arxiv/src/arXiv_src_1609_011.tar",
        "s3://arxiv/src/arXiv_src_1609_012.tar",
        "s3://arxiv/src/arXiv_src_1609_013.tar",
        "s3://arxiv/src/arXiv_src_1609_014.tar",
        "s3://arxiv/src/arXiv_src_1609_015.tar",
        "s3://arxiv/src/arXiv_src_1609_016.tar",
        "s3://arxiv/src/arXiv_src_1609_017.tar",
        "s3://arxiv/src/arXiv_src_1609_018.tar",
        "s3://arxiv/src/arXiv_src_1609_019.tar",
        "s3://arxiv/src/arXiv_src_1609_020.tar",
        "s3://arxiv/src/arXiv_src_1609_021.tar",
        "s3://arxiv/src/arXiv_src_1609_022.tar",
        "s3://arxiv/src/arXiv_src_1609_023.tar",
        "s3://arxiv/src/arXiv_src_1610_001.tar",
        "s3://arxiv/src/arXiv_src_1610_002.tar",
        "s3://arxiv/src/arXiv_src_1610_003.tar",
        "s3://arxiv/src/arXiv_src_1610_004.tar",
        "s3://arxiv/src/arXiv_src_1610_005.tar",
        "s3://arxiv/src/arXiv_src_1610_006.tar",
        "s3://arxiv/src/arXiv_src_1610_007.tar",
        "s3://arxiv/src/arXiv_src_1610_008.tar",
        "s3://arxiv/src/arXiv_src_1610_009.tar",
        "s3://arxiv/src/arXiv_src_1610_010.tar",
        "s3://arxiv/src/arXiv_src_1610_011.tar",
        "s3://arxiv/src/arXiv_src_1610_012.tar",
        "s3://arxiv/src/arXiv_src_1610_013.tar",
        "s3://arxiv/src/arXiv_src_1610_014.tar",
        "s3://arxiv/src/arXiv_src_1610_015.tar",
        "s3://arxiv/src/arXiv_src_1610_016.tar",
        "s3://arxiv/src/arXiv_src_1610_017.tar",
        "s3://arxiv/src/arXiv_src_1610_018.tar",
        "s3://arxiv/src/arXiv_src_1610_019.tar",
        "s3://arxiv/src/arXiv_src_1610_020.tar",
        "s3://arxiv/src/arXiv_src_1610_021.tar",
        "s3://arxiv/src/arXiv_src_1610_022.tar",
        "s3://arxiv/src/arXiv_src_1610_023.tar",
        "s3://arxiv/src/arXiv_src_1611_001.tar",
        "s3://arxiv/src/arXiv_src_1611_002.tar",
        "s3://arxiv/src/arXiv_src_1611_003.tar",
        "s3://arxiv/src/arXiv_src_1611_004.tar",
        "s3://arxiv/src/arXiv_src_1611_005.tar",
        "s3://arxiv/src/arXiv_src_1611_006.tar",
        "s3://arxiv/src/arXiv_src_1611_007.tar",
        "s3://arxiv/src/arXiv_src_1611_008.tar",
        "s3://arxiv/src/arXiv_src_1611_009.tar",
        "s3://arxiv/src/arXiv_src_1611_010.tar",
        "s3://arxiv/src/arXiv_src_1611_011.tar",
        "s3://arxiv/src/arXiv_src_1611_012.tar"
    ]
    # Process all papers simultaneously to avoid blocking on the ones
    # where pdflatex runs forever
    grouped_tarnames = figure_utils.ordered_group_by(
        tarnames, lambda x: True
    )
    for group_key, group_tars in grouped_tarnames.items():
        print(datetime.datetime.now())
        tmpdir = settings.ARXIV_DATA_TMP_DIR
        # with tempfile.TemporaryDirectory(
        #     prefix=settings.ARXIV_DATA_TMP_DIR
        # ) as tmpdir:
        tmpdir += '/'
        f = functools.partial(download_and_extract_tar, extract_dir=tmpdir)
        print(
            'Downloading %d tarfiles in group %s' %
            (len(group_tars), str(group_key))
        )
        with multiprocessing.Pool() as p:
            p.map(f, group_tars)
        # paper_tarnames = glob.glob(os.path.join(tmpdir, '*/*.gz'))
        # print(datetime.datetime.now())
        # print(
        #     'Processing %d papers in group %s' %
        #     (len(paper_tarnames), str(group_key))
        # )
        # with multiprocessing.Pool(processes=round(settings.PROCESS_PAPER_TAR_THREAD_COUNT)
        #                           ) as p:
        #     p.map(process_paper_tar, paper_tarnames)


if __name__ == "__main__":
    logging.basicConfig(filename='logger_arxiv.log', level=logging.WARNING)
    run_on_all()
    print('All done')
