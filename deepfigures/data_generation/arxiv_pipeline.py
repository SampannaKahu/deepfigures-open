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
        "s3://arxiv/src/arXiv_src_1611_013.tar",
        "s3://arxiv/src/arXiv_src_1611_014.tar",
        "s3://arxiv/src/arXiv_src_1611_015.tar",
        "s3://arxiv/src/arXiv_src_1611_016.tar",
        "s3://arxiv/src/arXiv_src_1611_017.tar",
        "s3://arxiv/src/arXiv_src_1611_018.tar",
        "s3://arxiv/src/arXiv_src_1611_019.tar",
        "s3://arxiv/src/arXiv_src_1611_020.tar",
        "s3://arxiv/src/arXiv_src_1611_021.tar",
        "s3://arxiv/src/arXiv_src_1611_022.tar",
        "s3://arxiv/src/arXiv_src_1611_023.tar",
        "s3://arxiv/src/arXiv_src_1611_024.tar",
        "s3://arxiv/src/arXiv_src_1612_001.tar",
        "s3://arxiv/src/arXiv_src_1612_002.tar",
        "s3://arxiv/src/arXiv_src_1612_003.tar",
        "s3://arxiv/src/arXiv_src_1612_004.tar",
        "s3://arxiv/src/arXiv_src_1612_005.tar",
        "s3://arxiv/src/arXiv_src_1612_006.tar",
        "s3://arxiv/src/arXiv_src_1612_007.tar",
        "s3://arxiv/src/arXiv_src_1612_008.tar",
        "s3://arxiv/src/arXiv_src_1612_009.tar",
        "s3://arxiv/src/arXiv_src_1612_010.tar",
        "s3://arxiv/src/arXiv_src_1612_011.tar",
        "s3://arxiv/src/arXiv_src_1612_012.tar",
        "s3://arxiv/src/arXiv_src_1612_013.tar",
        "s3://arxiv/src/arXiv_src_1612_014.tar",
        "s3://arxiv/src/arXiv_src_1612_015.tar",
        "s3://arxiv/src/arXiv_src_1612_016.tar",
        "s3://arxiv/src/arXiv_src_1612_017.tar",
        "s3://arxiv/src/arXiv_src_1612_018.tar",
        "s3://arxiv/src/arXiv_src_1612_019.tar",
        "s3://arxiv/src/arXiv_src_1612_020.tar",
        "s3://arxiv/src/arXiv_src_1612_021.tar",
        "s3://arxiv/src/arXiv_src_1612_022.tar",
        "s3://arxiv/src/arXiv_src_1701_001.tar",
        "s3://arxiv/src/arXiv_src_1701_002.tar",
        "s3://arxiv/src/arXiv_src_1701_003.tar",
        "s3://arxiv/src/arXiv_src_1701_004.tar",
        "s3://arxiv/src/arXiv_src_1701_005.tar",
        "s3://arxiv/src/arXiv_src_1701_006.tar",
        "s3://arxiv/src/arXiv_src_1701_007.tar",
        "s3://arxiv/src/arXiv_src_1701_008.tar",
        "s3://arxiv/src/arXiv_src_1701_009.tar",
        "s3://arxiv/src/arXiv_src_1701_010.tar",
        "s3://arxiv/src/arXiv_src_1701_011.tar",
        "s3://arxiv/src/arXiv_src_1701_012.tar",
        "s3://arxiv/src/arXiv_src_1701_013.tar",
        "s3://arxiv/src/arXiv_src_1701_014.tar",
        "s3://arxiv/src/arXiv_src_1701_015.tar",
        "s3://arxiv/src/arXiv_src_1701_016.tar",
        "s3://arxiv/src/arXiv_src_1701_017.tar",
        "s3://arxiv/src/arXiv_src_1701_018.tar",
        "s3://arxiv/src/arXiv_src_1701_019.tar",
        "s3://arxiv/src/arXiv_src_1701_020.tar",
        "s3://arxiv/src/arXiv_src_1701_021.tar",
        "s3://arxiv/src/arXiv_src_1702_001.tar",
        "s3://arxiv/src/arXiv_src_1702_002.tar",
        "s3://arxiv/src/arXiv_src_1702_003.tar",
        "s3://arxiv/src/arXiv_src_1702_004.tar",
        "s3://arxiv/src/arXiv_src_1702_005.tar",
        "s3://arxiv/src/arXiv_src_1702_006.tar",
        "s3://arxiv/src/arXiv_src_1702_007.tar",
        "s3://arxiv/src/arXiv_src_1702_008.tar",
        "s3://arxiv/src/arXiv_src_1702_009.tar",
        "s3://arxiv/src/arXiv_src_1702_010.tar",
        "s3://arxiv/src/arXiv_src_1702_011.tar",
        "s3://arxiv/src/arXiv_src_1702_012.tar",
        "s3://arxiv/src/arXiv_src_1702_013.tar",
        "s3://arxiv/src/arXiv_src_1702_014.tar",
        "s3://arxiv/src/arXiv_src_1702_015.tar",
        "s3://arxiv/src/arXiv_src_1702_016.tar",
        "s3://arxiv/src/arXiv_src_1702_017.tar",
        "s3://arxiv/src/arXiv_src_1702_018.tar",
        "s3://arxiv/src/arXiv_src_1702_019.tar",
        "s3://arxiv/src/arXiv_src_1702_020.tar",
        "s3://arxiv/src/arXiv_src_1703_001.tar",
        "s3://arxiv/src/arXiv_src_1703_002.tar",
        "s3://arxiv/src/arXiv_src_1703_003.tar",
        "s3://arxiv/src/arXiv_src_1703_004.tar",
        "s3://arxiv/src/arXiv_src_1703_005.tar",
        "s3://arxiv/src/arXiv_src_1703_006.tar",
        "s3://arxiv/src/arXiv_src_1703_007.tar",
        "s3://arxiv/src/arXiv_src_1703_008.tar",
        "s3://arxiv/src/arXiv_src_1703_009.tar",
        "s3://arxiv/src/arXiv_src_1703_010.tar",
        "s3://arxiv/src/arXiv_src_1703_011.tar",
        "s3://arxiv/src/arXiv_src_1703_012.tar",
        "s3://arxiv/src/arXiv_src_1703_013.tar",
        "s3://arxiv/src/arXiv_src_1703_014.tar",
        "s3://arxiv/src/arXiv_src_1703_015.tar",
        "s3://arxiv/src/arXiv_src_1703_016.tar",
        "s3://arxiv/src/arXiv_src_1703_017.tar",
        "s3://arxiv/src/arXiv_src_1703_018.tar",
        "s3://arxiv/src/arXiv_src_1703_019.tar",
        "s3://arxiv/src/arXiv_src_1703_020.tar",
        "s3://arxiv/src/arXiv_src_1703_021.tar",
        "s3://arxiv/src/arXiv_src_1703_022.tar",
        "s3://arxiv/src/arXiv_src_1703_023.tar",
        "s3://arxiv/src/arXiv_src_1703_024.tar",
        "s3://arxiv/src/arXiv_src_1703_025.tar",
        "s3://arxiv/src/arXiv_src_1703_026.tar",
        "s3://arxiv/src/arXiv_src_1704_001.tar",
        "s3://arxiv/src/arXiv_src_1704_002.tar",
        "s3://arxiv/src/arXiv_src_1704_003.tar",
        "s3://arxiv/src/arXiv_src_1704_004.tar",
        "s3://arxiv/src/arXiv_src_1704_005.tar",
        "s3://arxiv/src/arXiv_src_1704_006.tar",
        "s3://arxiv/src/arXiv_src_1704_007.tar",
        "s3://arxiv/src/arXiv_src_1704_008.tar",
        "s3://arxiv/src/arXiv_src_1704_009.tar",
        "s3://arxiv/src/arXiv_src_1704_010.tar",
        "s3://arxiv/src/arXiv_src_1704_011.tar",
        "s3://arxiv/src/arXiv_src_1704_012.tar",
        "s3://arxiv/src/arXiv_src_1704_013.tar",
        "s3://arxiv/src/arXiv_src_1704_014.tar",
        "s3://arxiv/src/arXiv_src_1704_015.tar",
        "s3://arxiv/src/arXiv_src_1704_016.tar",
        "s3://arxiv/src/arXiv_src_1704_017.tar",
        "s3://arxiv/src/arXiv_src_1704_018.tar",
        "s3://arxiv/src/arXiv_src_1704_019.tar",
        "s3://arxiv/src/arXiv_src_1704_020.tar",
        "s3://arxiv/src/arXiv_src_1704_021.tar",
        "s3://arxiv/src/arXiv_src_1704_022.tar",
        "s3://arxiv/src/arXiv_src_1704_023.tar",
        "s3://arxiv/src/arXiv_src_1705_001.tar",
        "s3://arxiv/src/arXiv_src_1705_002.tar",
        "s3://arxiv/src/arXiv_src_1705_003.tar",
        "s3://arxiv/src/arXiv_src_1705_004.tar",
        "s3://arxiv/src/arXiv_src_1705_005.tar",
        "s3://arxiv/src/arXiv_src_1705_006.tar",
        "s3://arxiv/src/arXiv_src_1705_007.tar",
        "s3://arxiv/src/arXiv_src_1705_008.tar",
        "s3://arxiv/src/arXiv_src_1705_009.tar",
        "s3://arxiv/src/arXiv_src_1705_010.tar",
        "s3://arxiv/src/arXiv_src_1705_011.tar",
        "s3://arxiv/src/arXiv_src_1705_012.tar",
        "s3://arxiv/src/arXiv_src_1705_013.tar",
        "s3://arxiv/src/arXiv_src_1705_014.tar",
        "s3://arxiv/src/arXiv_src_1705_015.tar",
        "s3://arxiv/src/arXiv_src_1705_016.tar",
        "s3://arxiv/src/arXiv_src_1705_017.tar",
        "s3://arxiv/src/arXiv_src_1705_018.tar",
        "s3://arxiv/src/arXiv_src_1705_019.tar",
        "s3://arxiv/src/arXiv_src_1705_020.tar",
        "s3://arxiv/src/arXiv_src_1705_021.tar",
        "s3://arxiv/src/arXiv_src_1705_022.tar",
        "s3://arxiv/src/arXiv_src_1705_023.tar",
        "s3://arxiv/src/arXiv_src_1705_024.tar",
        "s3://arxiv/src/arXiv_src_1705_025.tar",
        "s3://arxiv/src/arXiv_src_1705_026.tar",
        "s3://arxiv/src/arXiv_src_1705_027.tar",
        "s3://arxiv/src/arXiv_src_1706_001.tar",
        "s3://arxiv/src/arXiv_src_1706_002.tar",
        "s3://arxiv/src/arXiv_src_1706_003.tar",
        "s3://arxiv/src/arXiv_src_1706_004.tar",
        "s3://arxiv/src/arXiv_src_1706_005.tar",
        "s3://arxiv/src/arXiv_src_1706_006.tar",
        "s3://arxiv/src/arXiv_src_1706_007.tar",
        "s3://arxiv/src/arXiv_src_1706_008.tar",
        "s3://arxiv/src/arXiv_src_1706_009.tar",
        "s3://arxiv/src/arXiv_src_1706_010.tar",
        "s3://arxiv/src/arXiv_src_1706_011.tar",
        "s3://arxiv/src/arXiv_src_1706_012.tar",
        "s3://arxiv/src/arXiv_src_1706_013.tar",
        "s3://arxiv/src/arXiv_src_1706_014.tar",
        "s3://arxiv/src/arXiv_src_1706_015.tar",
        "s3://arxiv/src/arXiv_src_1706_016.tar",
        "s3://arxiv/src/arXiv_src_1706_017.tar",
        "s3://arxiv/src/arXiv_src_1706_018.tar",
        "s3://arxiv/src/arXiv_src_1706_019.tar",
        "s3://arxiv/src/arXiv_src_1706_020.tar",
        "s3://arxiv/src/arXiv_src_1706_021.tar",
        "s3://arxiv/src/arXiv_src_1706_022.tar",
        "s3://arxiv/src/arXiv_src_1706_023.tar",
        "s3://arxiv/src/arXiv_src_1706_024.tar",
        "s3://arxiv/src/arXiv_src_1706_025.tar",
        "s3://arxiv/src/arXiv_src_1706_026.tar",
        "s3://arxiv/src/arXiv_src_1707_001.tar",
        "s3://arxiv/src/arXiv_src_1707_002.tar",
        "s3://arxiv/src/arXiv_src_1707_003.tar",
        "s3://arxiv/src/arXiv_src_1707_004.tar",
        "s3://arxiv/src/arXiv_src_1707_005.tar",
        "s3://arxiv/src/arXiv_src_1707_006.tar",
        "s3://arxiv/src/arXiv_src_1707_007.tar",
        "s3://arxiv/src/arXiv_src_1707_008.tar",
        "s3://arxiv/src/arXiv_src_1707_009.tar",
        "s3://arxiv/src/arXiv_src_1707_010.tar",
        "s3://arxiv/src/arXiv_src_1707_011.tar",
        "s3://arxiv/src/arXiv_src_1707_012.tar",
        "s3://arxiv/src/arXiv_src_1707_013.tar",
        "s3://arxiv/src/arXiv_src_1707_014.tar",
        "s3://arxiv/src/arXiv_src_1707_015.tar",
        "s3://arxiv/src/arXiv_src_1707_016.tar",
        "s3://arxiv/src/arXiv_src_1707_017.tar",
        "s3://arxiv/src/arXiv_src_1707_018.tar",
        "s3://arxiv/src/arXiv_src_1707_019.tar",
        "s3://arxiv/src/arXiv_src_1707_020.tar",
        "s3://arxiv/src/arXiv_src_1707_021.tar",
        "s3://arxiv/src/arXiv_src_1707_022.tar",
        "s3://arxiv/src/arXiv_src_1707_023.tar",
        "s3://arxiv/src/arXiv_src_1707_024.tar",
        "s3://arxiv/src/arXiv_src_1708_001.tar",
        "s3://arxiv/src/arXiv_src_1708_002.tar",
        "s3://arxiv/src/arXiv_src_1708_003.tar",
        "s3://arxiv/src/arXiv_src_1708_004.tar",
        "s3://arxiv/src/arXiv_src_1708_005.tar",
        "s3://arxiv/src/arXiv_src_1708_006.tar",
        "s3://arxiv/src/arXiv_src_1708_007.tar",
        "s3://arxiv/src/arXiv_src_1708_008.tar",
        "s3://arxiv/src/arXiv_src_1708_009.tar",
        "s3://arxiv/src/arXiv_src_1708_010.tar",
        "s3://arxiv/src/arXiv_src_1708_011.tar",
        "s3://arxiv/src/arXiv_src_1708_012.tar",
        "s3://arxiv/src/arXiv_src_1708_013.tar",
        "s3://arxiv/src/arXiv_src_1708_014.tar",
        "s3://arxiv/src/arXiv_src_1708_015.tar",
        "s3://arxiv/src/arXiv_src_1708_016.tar",
        "s3://arxiv/src/arXiv_src_1708_017.tar",
        "s3://arxiv/src/arXiv_src_1708_018.tar",
        "s3://arxiv/src/arXiv_src_1708_019.tar",
        "s3://arxiv/src/arXiv_src_1708_020.tar",
        "s3://arxiv/src/arXiv_src_1708_021.tar",
        "s3://arxiv/src/arXiv_src_1708_022.tar",
        "s3://arxiv/src/arXiv_src_1708_023.tar",
        "s3://arxiv/src/arXiv_src_1708_024.tar",
        "s3://arxiv/src/arXiv_src_1708_025.tar",
        "s3://arxiv/src/arXiv_src_1709_001.tar",
        "s3://arxiv/src/arXiv_src_1709_002.tar",
        "s3://arxiv/src/arXiv_src_1709_003.tar",
        "s3://arxiv/src/arXiv_src_1709_004.tar",
        "s3://arxiv/src/arXiv_src_1709_005.tar",
        "s3://arxiv/src/arXiv_src_1709_006.tar",
        "s3://arxiv/src/arXiv_src_1709_007.tar",
        "s3://arxiv/src/arXiv_src_1709_008.tar",
        "s3://arxiv/src/arXiv_src_1709_009.tar",
        "s3://arxiv/src/arXiv_src_1709_010.tar",
        "s3://arxiv/src/arXiv_src_1709_011.tar",
        "s3://arxiv/src/arXiv_src_1709_012.tar",
        "s3://arxiv/src/arXiv_src_1709_013.tar",
        "s3://arxiv/src/arXiv_src_1709_014.tar",
        "s3://arxiv/src/arXiv_src_1709_015.tar",
        "s3://arxiv/src/arXiv_src_1709_016.tar",
        "s3://arxiv/src/arXiv_src_1709_017.tar",
        "s3://arxiv/src/arXiv_src_1709_018.tar",
        "s3://arxiv/src/arXiv_src_1709_019.tar",
        "s3://arxiv/src/arXiv_src_1709_020.tar",
        "s3://arxiv/src/arXiv_src_1709_021.tar",
        "s3://arxiv/src/arXiv_src_1709_022.tar",
        "s3://arxiv/src/arXiv_src_1709_023.tar",
        "s3://arxiv/src/arXiv_src_1709_024.tar",
        "s3://arxiv/src/arXiv_src_1709_025.tar",
        "s3://arxiv/src/arXiv_src_1709_026.tar",
        "s3://arxiv/src/arXiv_src_1709_027.tar",
        "s3://arxiv/src/arXiv_src_1710_001.tar",
        "s3://arxiv/src/arXiv_src_1710_002.tar",
        "s3://arxiv/src/arXiv_src_1710_003.tar",
        "s3://arxiv/src/arXiv_src_1710_004.tar",
        "s3://arxiv/src/arXiv_src_1710_005.tar",
        "s3://arxiv/src/arXiv_src_1710_006.tar",
        "s3://arxiv/src/arXiv_src_1710_007.tar",
        "s3://arxiv/src/arXiv_src_1710_008.tar",
        "s3://arxiv/src/arXiv_src_1710_009.tar",
        "s3://arxiv/src/arXiv_src_1710_010.tar",
        "s3://arxiv/src/arXiv_src_1710_011.tar",
        "s3://arxiv/src/arXiv_src_1710_012.tar",
        "s3://arxiv/src/arXiv_src_1710_013.tar",
        "s3://arxiv/src/arXiv_src_1710_014.tar",
        "s3://arxiv/src/arXiv_src_1710_015.tar",
        "s3://arxiv/src/arXiv_src_1710_016.tar",
        "s3://arxiv/src/arXiv_src_1710_017.tar",
        "s3://arxiv/src/arXiv_src_1710_018.tar",
        "s3://arxiv/src/arXiv_src_1710_019.tar",
        "s3://arxiv/src/arXiv_src_1710_020.tar",
        "s3://arxiv/src/arXiv_src_1710_021.tar",
        "s3://arxiv/src/arXiv_src_1710_022.tar",
        "s3://arxiv/src/arXiv_src_1710_023.tar",
        "s3://arxiv/src/arXiv_src_1710_024.tar",
        "s3://arxiv/src/arXiv_src_1710_025.tar",
        "s3://arxiv/src/arXiv_src_1710_026.tar",
        "s3://arxiv/src/arXiv_src_1710_027.tar",
        "s3://arxiv/src/arXiv_src_1710_028.tar",
        "s3://arxiv/src/arXiv_src_1710_029.tar",
        "s3://arxiv/src/arXiv_src_1711_001.tar",
        "s3://arxiv/src/arXiv_src_1711_002.tar",
        "s3://arxiv/src/arXiv_src_1711_003.tar",
        "s3://arxiv/src/arXiv_src_1711_004.tar",
        "s3://arxiv/src/arXiv_src_1711_005.tar",
        "s3://arxiv/src/arXiv_src_1711_006.tar",
        "s3://arxiv/src/arXiv_src_1711_007.tar",
        "s3://arxiv/src/arXiv_src_1711_008.tar",
        "s3://arxiv/src/arXiv_src_1711_009.tar",
        "s3://arxiv/src/arXiv_src_1711_010.tar",
        "s3://arxiv/src/arXiv_src_1711_011.tar",
        "s3://arxiv/src/arXiv_src_1711_012.tar",
        "s3://arxiv/src/arXiv_src_1711_013.tar",
        "s3://arxiv/src/arXiv_src_1711_014.tar",
        "s3://arxiv/src/arXiv_src_1711_015.tar",
        "s3://arxiv/src/arXiv_src_1711_016.tar",
        "s3://arxiv/src/arXiv_src_1711_017.tar",
        "s3://arxiv/src/arXiv_src_1711_018.tar",
        "s3://arxiv/src/arXiv_src_1711_019.tar",
        "s3://arxiv/src/arXiv_src_1711_020.tar",
        "s3://arxiv/src/arXiv_src_1711_021.tar",
        "s3://arxiv/src/arXiv_src_1711_022.tar",
        "s3://arxiv/src/arXiv_src_1711_023.tar",
        "s3://arxiv/src/arXiv_src_1711_024.tar",
        "s3://arxiv/src/arXiv_src_1711_025.tar",
        "s3://arxiv/src/arXiv_src_1711_026.tar",
        "s3://arxiv/src/arXiv_src_1711_027.tar",
        "s3://arxiv/src/arXiv_src_1711_028.tar",
        "s3://arxiv/src/arXiv_src_1711_029.tar",
        "s3://arxiv/src/arXiv_src_1711_030.tar",
        "s3://arxiv/src/arXiv_src_1712_001.tar",
        "s3://arxiv/src/arXiv_src_1712_002.tar",
        "s3://arxiv/src/arXiv_src_1712_003.tar",
        "s3://arxiv/src/arXiv_src_1712_004.tar",
        "s3://arxiv/src/arXiv_src_1712_005.tar",
        "s3://arxiv/src/arXiv_src_1712_006.tar",
        "s3://arxiv/src/arXiv_src_1712_007.tar",
        "s3://arxiv/src/arXiv_src_1712_008.tar",
        "s3://arxiv/src/arXiv_src_1712_009.tar",
        "s3://arxiv/src/arXiv_src_1712_010.tar",
        "s3://arxiv/src/arXiv_src_1712_011.tar",
        "s3://arxiv/src/arXiv_src_1712_012.tar",
        "s3://arxiv/src/arXiv_src_1712_013.tar",
        "s3://arxiv/src/arXiv_src_1712_014.tar",
        "s3://arxiv/src/arXiv_src_1712_015.tar",
        "s3://arxiv/src/arXiv_src_1712_016.tar",
        "s3://arxiv/src/arXiv_src_1712_017.tar",
        "s3://arxiv/src/arXiv_src_1712_018.tar",
        "s3://arxiv/src/arXiv_src_1712_019.tar",
        "s3://arxiv/src/arXiv_src_1712_020.tar",
        "s3://arxiv/src/arXiv_src_1712_021.tar",
        "s3://arxiv/src/arXiv_src_1712_022.tar",
        "s3://arxiv/src/arXiv_src_1712_023.tar",
        "s3://arxiv/src/arXiv_src_1712_024.tar",
        "s3://arxiv/src/arXiv_src_1712_025.tar",
        "s3://arxiv/src/arXiv_src_1712_026.tar",
        "s3://arxiv/src/arXiv_src_1712_027.tar",
        "s3://arxiv/src/arXiv_src_1801_001.tar",
        "s3://arxiv/src/arXiv_src_1801_002.tar",
        "s3://arxiv/src/arXiv_src_1801_003.tar",
        "s3://arxiv/src/arXiv_src_1801_004.tar",
        "s3://arxiv/src/arXiv_src_1801_005.tar",
        "s3://arxiv/src/arXiv_src_1801_006.tar",
        "s3://arxiv/src/arXiv_src_1801_007.tar",
        "s3://arxiv/src/arXiv_src_1801_008.tar",
        "s3://arxiv/src/arXiv_src_1801_009.tar",
        "s3://arxiv/src/arXiv_src_1801_010.tar",
        "s3://arxiv/src/arXiv_src_1801_011.tar",
        "s3://arxiv/src/arXiv_src_1801_012.tar",
        "s3://arxiv/src/arXiv_src_1801_013.tar",
        "s3://arxiv/src/arXiv_src_1801_014.tar",
        "s3://arxiv/src/arXiv_src_1801_015.tar",
        "s3://arxiv/src/arXiv_src_1801_016.tar",
        "s3://arxiv/src/arXiv_src_1801_017.tar",
        "s3://arxiv/src/arXiv_src_1801_018.tar",
        "s3://arxiv/src/arXiv_src_1801_019.tar",
        "s3://arxiv/src/arXiv_src_1801_020.tar",
        "s3://arxiv/src/arXiv_src_1801_021.tar",
        "s3://arxiv/src/arXiv_src_1801_022.tar",
        "s3://arxiv/src/arXiv_src_1801_023.tar",
        "s3://arxiv/src/arXiv_src_1801_024.tar",
        "s3://arxiv/src/arXiv_src_1801_025.tar",
        "s3://arxiv/src/arXiv_src_1801_026.tar",
        "s3://arxiv/src/arXiv_src_1801_027.tar",
        "s3://arxiv/src/arXiv_src_1802_001.tar",
        "s3://arxiv/src/arXiv_src_1802_002.tar",
        "s3://arxiv/src/arXiv_src_1802_003.tar",
        "s3://arxiv/src/arXiv_src_1802_004.tar",
        "s3://arxiv/src/arXiv_src_1802_005.tar",
        "s3://arxiv/src/arXiv_src_1802_006.tar",
        "s3://arxiv/src/arXiv_src_1802_007.tar",
        "s3://arxiv/src/arXiv_src_1802_008.tar",
        "s3://arxiv/src/arXiv_src_1802_009.tar",
        "s3://arxiv/src/arXiv_src_1802_010.tar",
        "s3://arxiv/src/arXiv_src_1802_011.tar",
        "s3://arxiv/src/arXiv_src_1802_012.tar",
        "s3://arxiv/src/arXiv_src_1802_013.tar",
        "s3://arxiv/src/arXiv_src_1802_014.tar",
        "s3://arxiv/src/arXiv_src_1802_015.tar",
        "s3://arxiv/src/arXiv_src_1802_016.tar",
        "s3://arxiv/src/arXiv_src_1802_017.tar",
        "s3://arxiv/src/arXiv_src_1802_018.tar",
        "s3://arxiv/src/arXiv_src_1802_019.tar",
        "s3://arxiv/src/arXiv_src_1802_020.tar",
        "s3://arxiv/src/arXiv_src_1802_021.tar",
        "s3://arxiv/src/arXiv_src_1802_022.tar",
        "s3://arxiv/src/arXiv_src_1802_023.tar",
        "s3://arxiv/src/arXiv_src_1802_024.tar",
        "s3://arxiv/src/arXiv_src_1802_025.tar",
        "s3://arxiv/src/arXiv_src_1802_026.tar",
        "s3://arxiv/src/arXiv_src_1802_027.tar",
        "s3://arxiv/src/arXiv_src_1802_028.tar",
        "s3://arxiv/src/arXiv_src_1803_001.tar",
        "s3://arxiv/src/arXiv_src_1803_002.tar",
        "s3://arxiv/src/arXiv_src_1803_003.tar",
        "s3://arxiv/src/arXiv_src_1803_004.tar",
        "s3://arxiv/src/arXiv_src_1803_005.tar",
        "s3://arxiv/src/arXiv_src_1803_006.tar",
        "s3://arxiv/src/arXiv_src_1803_007.tar",
        "s3://arxiv/src/arXiv_src_1803_008.tar",
        "s3://arxiv/src/arXiv_src_1803_009.tar",
        "s3://arxiv/src/arXiv_src_1803_010.tar",
        "s3://arxiv/src/arXiv_src_1803_011.tar",
        "s3://arxiv/src/arXiv_src_1803_012.tar",
        "s3://arxiv/src/arXiv_src_1803_013.tar",
        "s3://arxiv/src/arXiv_src_1803_014.tar",
        "s3://arxiv/src/arXiv_src_1803_015.tar",
        "s3://arxiv/src/arXiv_src_1803_016.tar",
        "s3://arxiv/src/arXiv_src_1803_017.tar",
        "s3://arxiv/src/arXiv_src_1803_018.tar",
        "s3://arxiv/src/arXiv_src_1803_019.tar",
        "s3://arxiv/src/arXiv_src_1803_020.tar",
        "s3://arxiv/src/arXiv_src_1803_021.tar",
        "s3://arxiv/src/arXiv_src_1803_022.tar",
        "s3://arxiv/src/arXiv_src_1803_023.tar",
        "s3://arxiv/src/arXiv_src_1803_024.tar",
        "s3://arxiv/src/arXiv_src_1803_025.tar",
        "s3://arxiv/src/arXiv_src_1803_026.tar",
        "s3://arxiv/src/arXiv_src_1803_027.tar",
        "s3://arxiv/src/arXiv_src_1803_028.tar",
        "s3://arxiv/src/arXiv_src_1803_029.tar",
        "s3://arxiv/src/arXiv_src_1803_030.tar",
        "s3://arxiv/src/arXiv_src_1803_031.tar",
        "s3://arxiv/src/arXiv_src_1803_032.tar",
        "s3://arxiv/src/arXiv_src_1804_001.tar",
        "s3://arxiv/src/arXiv_src_1804_002.tar",
        "s3://arxiv/src/arXiv_src_1804_003.tar",
        "s3://arxiv/src/arXiv_src_1804_004.tar",
        "s3://arxiv/src/arXiv_src_1804_005.tar",
        "s3://arxiv/src/arXiv_src_1804_006.tar",
        "s3://arxiv/src/arXiv_src_1804_007.tar",
        "s3://arxiv/src/arXiv_src_1804_008.tar",
        "s3://arxiv/src/arXiv_src_1804_009.tar",
        "s3://arxiv/src/arXiv_src_1804_010.tar",
        "s3://arxiv/src/arXiv_src_1804_011.tar",
        "s3://arxiv/src/arXiv_src_1804_012.tar",
        "s3://arxiv/src/arXiv_src_1804_013.tar",
        "s3://arxiv/src/arXiv_src_1804_014.tar",
        "s3://arxiv/src/arXiv_src_1804_015.tar",
        "s3://arxiv/src/arXiv_src_1804_016.tar",
        "s3://arxiv/src/arXiv_src_1804_017.tar",
        "s3://arxiv/src/arXiv_src_1804_018.tar",
        "s3://arxiv/src/arXiv_src_1804_019.tar",
        "s3://arxiv/src/arXiv_src_1804_020.tar",
        "s3://arxiv/src/arXiv_src_1804_021.tar",
        "s3://arxiv/src/arXiv_src_1804_022.tar",
        "s3://arxiv/src/arXiv_src_1804_023.tar",
        "s3://arxiv/src/arXiv_src_1804_024.tar",
        "s3://arxiv/src/arXiv_src_1804_025.tar",
        "s3://arxiv/src/arXiv_src_1804_026.tar",
        "s3://arxiv/src/arXiv_src_1804_027.tar",
        "s3://arxiv/src/arXiv_src_1804_028.tar",
        "s3://arxiv/src/arXiv_src_1804_029.tar",
        "s3://arxiv/src/arXiv_src_1804_030.tar",
        "s3://arxiv/src/arXiv_src_1804_031.tar",
        "s3://arxiv/src/arXiv_src_1804_032.tar",
        "s3://arxiv/src/arXiv_src_1805_001.tar",
        "s3://arxiv/src/arXiv_src_1805_002.tar",
        "s3://arxiv/src/arXiv_src_1805_003.tar",
        "s3://arxiv/src/arXiv_src_1805_004.tar",
        "s3://arxiv/src/arXiv_src_1805_005.tar",
        "s3://arxiv/src/arXiv_src_1805_006.tar",
        "s3://arxiv/src/arXiv_src_1805_007.tar",
        "s3://arxiv/src/arXiv_src_1805_008.tar",
        "s3://arxiv/src/arXiv_src_1805_009.tar",
        "s3://arxiv/src/arXiv_src_1805_010.tar",
        "s3://arxiv/src/arXiv_src_1805_011.tar",
        "s3://arxiv/src/arXiv_src_1805_012.tar",
        "s3://arxiv/src/arXiv_src_1805_013.tar",
        "s3://arxiv/src/arXiv_src_1805_014.tar",
        "s3://arxiv/src/arXiv_src_1805_015.tar",
        "s3://arxiv/src/arXiv_src_1805_016.tar",
        "s3://arxiv/src/arXiv_src_1805_017.tar",
        "s3://arxiv/src/arXiv_src_1805_018.tar",
        "s3://arxiv/src/arXiv_src_1805_019.tar",
        "s3://arxiv/src/arXiv_src_1805_020.tar",
        "s3://arxiv/src/arXiv_src_1805_021.tar",
        "s3://arxiv/src/arXiv_src_1805_022.tar",
        "s3://arxiv/src/arXiv_src_1805_023.tar",
        "s3://arxiv/src/arXiv_src_1805_024.tar",
        "s3://arxiv/src/arXiv_src_1805_025.tar",
        "s3://arxiv/src/arXiv_src_1805_026.tar",
        "s3://arxiv/src/arXiv_src_1805_027.tar",
        "s3://arxiv/src/arXiv_src_1805_028.tar",
        "s3://arxiv/src/arXiv_src_1805_029.tar",
        "s3://arxiv/src/arXiv_src_1805_030.tar",
        "s3://arxiv/src/arXiv_src_1805_031.tar",
        "s3://arxiv/src/arXiv_src_1805_032.tar",
        "s3://arxiv/src/arXiv_src_1805_033.tar",
        "s3://arxiv/src/arXiv_src_1805_034.tar",
        "s3://arxiv/src/arXiv_src_1806_001.tar",
        "s3://arxiv/src/arXiv_src_1806_002.tar",
        "s3://arxiv/src/arXiv_src_1806_003.tar",
        "s3://arxiv/src/arXiv_src_1806_004.tar",
        "s3://arxiv/src/arXiv_src_1806_005.tar",
        "s3://arxiv/src/arXiv_src_1806_006.tar",
        "s3://arxiv/src/arXiv_src_1806_007.tar",
        "s3://arxiv/src/arXiv_src_1806_008.tar",
        "s3://arxiv/src/arXiv_src_1806_009.tar",
        "s3://arxiv/src/arXiv_src_1806_010.tar",
        "s3://arxiv/src/arXiv_src_1806_011.tar",
        "s3://arxiv/src/arXiv_src_1806_012.tar",
        "s3://arxiv/src/arXiv_src_1806_013.tar",
        "s3://arxiv/src/arXiv_src_1806_014.tar",
        "s3://arxiv/src/arXiv_src_1806_015.tar",
        "s3://arxiv/src/arXiv_src_1806_016.tar",
        "s3://arxiv/src/arXiv_src_1806_017.tar",
        "s3://arxiv/src/arXiv_src_1806_018.tar",
        "s3://arxiv/src/arXiv_src_1806_019.tar",
        "s3://arxiv/src/arXiv_src_1806_020.tar",
        "s3://arxiv/src/arXiv_src_1806_021.tar",
        "s3://arxiv/src/arXiv_src_1806_022.tar",
        "s3://arxiv/src/arXiv_src_1806_023.tar",
        "s3://arxiv/src/arXiv_src_1806_024.tar",
        "s3://arxiv/src/arXiv_src_1806_025.tar",
        "s3://arxiv/src/arXiv_src_1806_026.tar",
        "s3://arxiv/src/arXiv_src_1806_027.tar",
        "s3://arxiv/src/arXiv_src_1806_028.tar",
        "s3://arxiv/src/arXiv_src_1806_029.tar",
        "s3://arxiv/src/arXiv_src_1806_030.tar",
        "s3://arxiv/src/arXiv_src_1806_031.tar",
        "s3://arxiv/src/arXiv_src_1806_032.tar",
        "s3://arxiv/src/arXiv_src_1806_033.tar",
        "s3://arxiv/src/arXiv_src_1807_001.tar",
        "s3://arxiv/src/arXiv_src_1807_002.tar",
        "s3://arxiv/src/arXiv_src_1807_003.tar",
        "s3://arxiv/src/arXiv_src_1807_004.tar",
        "s3://arxiv/src/arXiv_src_1807_005.tar",
        "s3://arxiv/src/arXiv_src_1807_006.tar",
        "s3://arxiv/src/arXiv_src_1807_007.tar",
        "s3://arxiv/src/arXiv_src_1807_008.tar",
        "s3://arxiv/src/arXiv_src_1807_009.tar",
        "s3://arxiv/src/arXiv_src_1807_010.tar",
        "s3://arxiv/src/arXiv_src_1807_011.tar",
        "s3://arxiv/src/arXiv_src_1807_012.tar",
        "s3://arxiv/src/arXiv_src_1807_013.tar",
        "s3://arxiv/src/arXiv_src_1807_014.tar",
        "s3://arxiv/src/arXiv_src_1807_015.tar",
        "s3://arxiv/src/arXiv_src_1807_016.tar",
        "s3://arxiv/src/arXiv_src_1807_017.tar",
        "s3://arxiv/src/arXiv_src_1807_018.tar",
        "s3://arxiv/src/arXiv_src_1807_019.tar",
        "s3://arxiv/src/arXiv_src_1807_020.tar",
        "s3://arxiv/src/arXiv_src_1807_021.tar",
        "s3://arxiv/src/arXiv_src_1807_022.tar",
        "s3://arxiv/src/arXiv_src_1807_023.tar",
        "s3://arxiv/src/arXiv_src_1807_024.tar",
        "s3://arxiv/src/arXiv_src_1807_025.tar",
        "s3://arxiv/src/arXiv_src_1807_026.tar",
        "s3://arxiv/src/arXiv_src_1807_027.tar",
        "s3://arxiv/src/arXiv_src_1807_028.tar",
        "s3://arxiv/src/arXiv_src_1807_029.tar",
        "s3://arxiv/src/arXiv_src_1807_030.tar",
        "s3://arxiv/src/arXiv_src_1807_031.tar",
        "s3://arxiv/src/arXiv_src_1807_032.tar",
        "s3://arxiv/src/arXiv_src_1807_033.tar",
        "s3://arxiv/src/arXiv_src_1807_034.tar",
        "s3://arxiv/src/arXiv_src_1808_001.tar",
        "s3://arxiv/src/arXiv_src_1808_002.tar",
        "s3://arxiv/src/arXiv_src_1808_003.tar",
        "s3://arxiv/src/arXiv_src_1808_004.tar",
        "s3://arxiv/src/arXiv_src_1808_005.tar",
        "s3://arxiv/src/arXiv_src_1808_006.tar",
        "s3://arxiv/src/arXiv_src_1808_007.tar",
        "s3://arxiv/src/arXiv_src_1808_008.tar",
        "s3://arxiv/src/arXiv_src_1808_009.tar",
        "s3://arxiv/src/arXiv_src_1808_010.tar",
        "s3://arxiv/src/arXiv_src_1808_011.tar",
        "s3://arxiv/src/arXiv_src_1808_012.tar",
        "s3://arxiv/src/arXiv_src_1808_013.tar",
        "s3://arxiv/src/arXiv_src_1808_014.tar",
        "s3://arxiv/src/arXiv_src_1808_015.tar",
        "s3://arxiv/src/arXiv_src_1808_016.tar",
        "s3://arxiv/src/arXiv_src_1808_017.tar",
        "s3://arxiv/src/arXiv_src_1808_018.tar",
        "s3://arxiv/src/arXiv_src_1808_019.tar",
        "s3://arxiv/src/arXiv_src_1808_020.tar",
        "s3://arxiv/src/arXiv_src_1808_021.tar",
        "s3://arxiv/src/arXiv_src_1808_022.tar",
        "s3://arxiv/src/arXiv_src_1808_023.tar",
        "s3://arxiv/src/arXiv_src_1808_024.tar",
        "s3://arxiv/src/arXiv_src_1808_025.tar",
        "s3://arxiv/src/arXiv_src_1808_026.tar",
        "s3://arxiv/src/arXiv_src_1808_027.tar",
        "s3://arxiv/src/arXiv_src_1808_028.tar",
        "s3://arxiv/src/arXiv_src_1808_029.tar",
        "s3://arxiv/src/arXiv_src_1808_030.tar",
        "s3://arxiv/src/arXiv_src_1808_031.tar",
        "s3://arxiv/src/arXiv_src_1809_001.tar",
        "s3://arxiv/src/arXiv_src_1809_002.tar",
        "s3://arxiv/src/arXiv_src_1809_003.tar",
        "s3://arxiv/src/arXiv_src_1809_004.tar",
        "s3://arxiv/src/arXiv_src_1809_005.tar",
        "s3://arxiv/src/arXiv_src_1809_006.tar",
        "s3://arxiv/src/arXiv_src_1809_007.tar",
        "s3://arxiv/src/arXiv_src_1809_008.tar",
        "s3://arxiv/src/arXiv_src_1809_009.tar",
        "s3://arxiv/src/arXiv_src_1809_010.tar",
        "s3://arxiv/src/arXiv_src_1809_011.tar",
        "s3://arxiv/src/arXiv_src_1809_012.tar",
        "s3://arxiv/src/arXiv_src_1809_013.tar",
        "s3://arxiv/src/arXiv_src_1809_014.tar",
        "s3://arxiv/src/arXiv_src_1809_015.tar",
        "s3://arxiv/src/arXiv_src_1809_016.tar",
        "s3://arxiv/src/arXiv_src_1809_017.tar",
        "s3://arxiv/src/arXiv_src_1809_018.tar",
        "s3://arxiv/src/arXiv_src_1809_019.tar",
        "s3://arxiv/src/arXiv_src_1809_020.tar",
        "s3://arxiv/src/arXiv_src_1809_021.tar",
        "s3://arxiv/src/arXiv_src_1809_022.tar",
        "s3://arxiv/src/arXiv_src_1809_023.tar",
        "s3://arxiv/src/arXiv_src_1809_024.tar",
        "s3://arxiv/src/arXiv_src_1809_025.tar",
        "s3://arxiv/src/arXiv_src_1809_026.tar",
        "s3://arxiv/src/arXiv_src_1809_027.tar",
        "s3://arxiv/src/arXiv_src_1809_028.tar",
        "s3://arxiv/src/arXiv_src_1809_029.tar",
        "s3://arxiv/src/arXiv_src_1809_030.tar",
        "s3://arxiv/src/arXiv_src_1809_031.tar",
        "s3://arxiv/src/arXiv_src_1809_032.tar",
        "s3://arxiv/src/arXiv_src_1810_001.tar",
        "s3://arxiv/src/arXiv_src_1810_002.tar",
        "s3://arxiv/src/arXiv_src_1810_003.tar",
        "s3://arxiv/src/arXiv_src_1810_004.tar",
        "s3://arxiv/src/arXiv_src_1810_005.tar",
        "s3://arxiv/src/arXiv_src_1810_006.tar",
        "s3://arxiv/src/arXiv_src_1810_007.tar",
        "s3://arxiv/src/arXiv_src_1810_008.tar",
        "s3://arxiv/src/arXiv_src_1810_009.tar",
        "s3://arxiv/src/arXiv_src_1810_010.tar",
        "s3://arxiv/src/arXiv_src_1810_011.tar",
        "s3://arxiv/src/arXiv_src_1810_012.tar",
        "s3://arxiv/src/arXiv_src_1810_013.tar",
        "s3://arxiv/src/arXiv_src_1810_014.tar",
        "s3://arxiv/src/arXiv_src_1810_015.tar",
        "s3://arxiv/src/arXiv_src_1810_016.tar",
        "s3://arxiv/src/arXiv_src_1810_017.tar",
        "s3://arxiv/src/arXiv_src_1810_018.tar",
        "s3://arxiv/src/arXiv_src_1810_019.tar",
        "s3://arxiv/src/arXiv_src_1810_020.tar",
        "s3://arxiv/src/arXiv_src_1810_021.tar",
        "s3://arxiv/src/arXiv_src_1810_022.tar",
        "s3://arxiv/src/arXiv_src_1810_023.tar",
        "s3://arxiv/src/arXiv_src_1810_024.tar",
        "s3://arxiv/src/arXiv_src_1810_025.tar",
        "s3://arxiv/src/arXiv_src_1810_026.tar",
        "s3://arxiv/src/arXiv_src_1810_027.tar",
        "s3://arxiv/src/arXiv_src_1810_028.tar",
        "s3://arxiv/src/arXiv_src_1810_029.tar",
        "s3://arxiv/src/arXiv_src_1810_030.tar",
        "s3://arxiv/src/arXiv_src_1810_031.tar",
        "s3://arxiv/src/arXiv_src_1810_032.tar",
        "s3://arxiv/src/arXiv_src_1810_033.tar",
        "s3://arxiv/src/arXiv_src_1810_034.tar",
        "s3://arxiv/src/arXiv_src_1810_035.tar",
        "s3://arxiv/src/arXiv_src_1810_036.tar",
        "s3://arxiv/src/arXiv_src_1810_037.tar",
        "s3://arxiv/src/arXiv_src_1811_001.tar",
        "s3://arxiv/src/arXiv_src_1811_002.tar",
        "s3://arxiv/src/arXiv_src_1811_003.tar",
        "s3://arxiv/src/arXiv_src_1811_004.tar",
        "s3://arxiv/src/arXiv_src_1811_005.tar",
        "s3://arxiv/src/arXiv_src_1811_006.tar",
        "s3://arxiv/src/arXiv_src_1811_007.tar",
        "s3://arxiv/src/arXiv_src_1811_008.tar",
        "s3://arxiv/src/arXiv_src_1811_009.tar",
        "s3://arxiv/src/arXiv_src_1811_010.tar",
        "s3://arxiv/src/arXiv_src_1811_011.tar",
        "s3://arxiv/src/arXiv_src_1811_012.tar",
        "s3://arxiv/src/arXiv_src_1811_013.tar",
        "s3://arxiv/src/arXiv_src_1811_014.tar",
        "s3://arxiv/src/arXiv_src_1811_015.tar",
        "s3://arxiv/src/arXiv_src_1811_016.tar",
        "s3://arxiv/src/arXiv_src_1811_017.tar",
        "s3://arxiv/src/arXiv_src_1811_018.tar",
        "s3://arxiv/src/arXiv_src_1811_019.tar",
        "s3://arxiv/src/arXiv_src_1811_020.tar",
        "s3://arxiv/src/arXiv_src_1811_021.tar",
        "s3://arxiv/src/arXiv_src_1811_022.tar",
        "s3://arxiv/src/arXiv_src_1811_023.tar",
        "s3://arxiv/src/arXiv_src_1811_024.tar",
        "s3://arxiv/src/arXiv_src_1811_025.tar",
        "s3://arxiv/src/arXiv_src_1811_026.tar",
        "s3://arxiv/src/arXiv_src_1811_027.tar",
        "s3://arxiv/src/arXiv_src_1811_028.tar",
        "s3://arxiv/src/arXiv_src_1811_029.tar",
        "s3://arxiv/src/arXiv_src_1811_030.tar",
        "s3://arxiv/src/arXiv_src_1811_031.tar",
        "s3://arxiv/src/arXiv_src_1811_032.tar",
        "s3://arxiv/src/arXiv_src_1811_033.tar",
        "s3://arxiv/src/arXiv_src_1811_034.tar",
        "s3://arxiv/src/arXiv_src_1811_035.tar",
        "s3://arxiv/src/arXiv_src_1811_036.tar",
        "s3://arxiv/src/arXiv_src_1811_037.tar",
        "s3://arxiv/src/arXiv_src_1812_001.tar",
        "s3://arxiv/src/arXiv_src_1812_002.tar",
        "s3://arxiv/src/arXiv_src_1812_003.tar",
        "s3://arxiv/src/arXiv_src_1812_004.tar",
        "s3://arxiv/src/arXiv_src_1812_005.tar",
        "s3://arxiv/src/arXiv_src_1812_006.tar",
        "s3://arxiv/src/arXiv_src_1812_007.tar",
        "s3://arxiv/src/arXiv_src_1812_008.tar",
        "s3://arxiv/src/arXiv_src_1812_009.tar",
        "s3://arxiv/src/arXiv_src_1812_010.tar",
        "s3://arxiv/src/arXiv_src_1812_011.tar",
        "s3://arxiv/src/arXiv_src_1812_012.tar",
        "s3://arxiv/src/arXiv_src_1812_013.tar",
        "s3://arxiv/src/arXiv_src_1812_014.tar",
        "s3://arxiv/src/arXiv_src_1812_015.tar",
        "s3://arxiv/src/arXiv_src_1812_016.tar",
        "s3://arxiv/src/arXiv_src_1812_017.tar",
        "s3://arxiv/src/arXiv_src_1812_018.tar",
        "s3://arxiv/src/arXiv_src_1812_019.tar",
        "s3://arxiv/src/arXiv_src_1812_020.tar",
        "s3://arxiv/src/arXiv_src_1812_021.tar",
        "s3://arxiv/src/arXiv_src_1812_022.tar",
        "s3://arxiv/src/arXiv_src_1812_023.tar",
        "s3://arxiv/src/arXiv_src_1812_024.tar",
        "s3://arxiv/src/arXiv_src_1812_025.tar",
        "s3://arxiv/src/arXiv_src_1812_026.tar",
        "s3://arxiv/src/arXiv_src_1812_027.tar",
        "s3://arxiv/src/arXiv_src_1812_028.tar",
        "s3://arxiv/src/arXiv_src_1812_029.tar",
        "s3://arxiv/src/arXiv_src_1812_030.tar",
        "s3://arxiv/src/arXiv_src_1812_031.tar",
        "s3://arxiv/src/arXiv_src_1812_032.tar",
        "s3://arxiv/src/arXiv_src_1812_033.tar",
        "s3://arxiv/src/arXiv_src_1812_034.tar",
        "s3://arxiv/src/arXiv_src_1812_035.tar",
        "s3://arxiv/src/arXiv_src_1901_001.tar",
        "s3://arxiv/src/arXiv_src_1901_002.tar",
        "s3://arxiv/src/arXiv_src_1901_003.tar",
        "s3://arxiv/src/arXiv_src_1901_004.tar",
        "s3://arxiv/src/arXiv_src_1901_005.tar",
        "s3://arxiv/src/arXiv_src_1901_006.tar",
        "s3://arxiv/src/arXiv_src_1901_007.tar",
        "s3://arxiv/src/arXiv_src_1901_008.tar",
        "s3://arxiv/src/arXiv_src_1901_009.tar",
        "s3://arxiv/src/arXiv_src_1901_010.tar",
        "s3://arxiv/src/arXiv_src_1901_011.tar",
        "s3://arxiv/src/arXiv_src_1901_012.tar",
        "s3://arxiv/src/arXiv_src_1901_013.tar",
        "s3://arxiv/src/arXiv_src_1901_014.tar",
        "s3://arxiv/src/arXiv_src_1901_015.tar",
        "s3://arxiv/src/arXiv_src_1901_016.tar",
        "s3://arxiv/src/arXiv_src_1901_017.tar",
        "s3://arxiv/src/arXiv_src_1901_018.tar",
        "s3://arxiv/src/arXiv_src_1901_019.tar",
        "s3://arxiv/src/arXiv_src_1901_020.tar",
        "s3://arxiv/src/arXiv_src_1901_021.tar",
        "s3://arxiv/src/arXiv_src_1901_022.tar",
        "s3://arxiv/src/arXiv_src_1901_023.tar",
        "s3://arxiv/src/arXiv_src_1901_024.tar",
        "s3://arxiv/src/arXiv_src_1901_025.tar",
        "s3://arxiv/src/arXiv_src_1901_026.tar",
        "s3://arxiv/src/arXiv_src_1901_027.tar",
        "s3://arxiv/src/arXiv_src_1901_028.tar",
        "s3://arxiv/src/arXiv_src_1901_029.tar",
        "s3://arxiv/src/arXiv_src_1901_030.tar",
        "s3://arxiv/src/arXiv_src_1901_031.tar",
        "s3://arxiv/src/arXiv_src_1901_032.tar",
        "s3://arxiv/src/arXiv_src_1901_033.tar",
        "s3://arxiv/src/arXiv_src_1902_001.tar",
        "s3://arxiv/src/arXiv_src_1902_002.tar",
        "s3://arxiv/src/arXiv_src_1902_003.tar",
        "s3://arxiv/src/arXiv_src_1902_004.tar",
        "s3://arxiv/src/arXiv_src_1902_005.tar",
        "s3://arxiv/src/arXiv_src_1902_006.tar",
        "s3://arxiv/src/arXiv_src_1902_007.tar",
        "s3://arxiv/src/arXiv_src_1902_008.tar",
        "s3://arxiv/src/arXiv_src_1902_009.tar",
        "s3://arxiv/src/arXiv_src_1902_010.tar",
        "s3://arxiv/src/arXiv_src_1902_011.tar",
        "s3://arxiv/src/arXiv_src_1902_012.tar",
        "s3://arxiv/src/arXiv_src_1902_013.tar",
        "s3://arxiv/src/arXiv_src_1902_014.tar",
        "s3://arxiv/src/arXiv_src_1902_015.tar",
        "s3://arxiv/src/arXiv_src_1902_016.tar",
        "s3://arxiv/src/arXiv_src_1902_017.tar",
        "s3://arxiv/src/arXiv_src_1902_018.tar",
        "s3://arxiv/src/arXiv_src_1902_019.tar",
        "s3://arxiv/src/arXiv_src_1902_020.tar",
        "s3://arxiv/src/arXiv_src_1902_021.tar",
        "s3://arxiv/src/arXiv_src_1902_022.tar",
        "s3://arxiv/src/arXiv_src_1902_023.tar",
        "s3://arxiv/src/arXiv_src_1902_024.tar",
        "s3://arxiv/src/arXiv_src_1902_025.tar",
        "s3://arxiv/src/arXiv_src_1902_026.tar",
        "s3://arxiv/src/arXiv_src_1902_027.tar",
        "s3://arxiv/src/arXiv_src_1902_028.tar",
        "s3://arxiv/src/arXiv_src_1902_029.tar",
        "s3://arxiv/src/arXiv_src_1902_030.tar",
        "s3://arxiv/src/arXiv_src_1902_031.tar",
        "s3://arxiv/src/arXiv_src_1902_032.tar",
        "s3://arxiv/src/arXiv_src_1902_033.tar",
        "s3://arxiv/src/arXiv_src_1903_001.tar",
        "s3://arxiv/src/arXiv_src_1903_002.tar",
        "s3://arxiv/src/arXiv_src_1903_003.tar",
        "s3://arxiv/src/arXiv_src_1903_004.tar",
        "s3://arxiv/src/arXiv_src_1903_005.tar",
        "s3://arxiv/src/arXiv_src_1903_006.tar",
        "s3://arxiv/src/arXiv_src_1903_007.tar",
        "s3://arxiv/src/arXiv_src_1903_008.tar",
        "s3://arxiv/src/arXiv_src_1903_009.tar",
        "s3://arxiv/src/arXiv_src_1903_010.tar",
        "s3://arxiv/src/arXiv_src_1903_011.tar",
        "s3://arxiv/src/arXiv_src_1903_012.tar",
        "s3://arxiv/src/arXiv_src_1903_013.tar",
        "s3://arxiv/src/arXiv_src_1903_014.tar",
        "s3://arxiv/src/arXiv_src_1903_015.tar",
        "s3://arxiv/src/arXiv_src_1903_016.tar",
        "s3://arxiv/src/arXiv_src_1903_017.tar",
        "s3://arxiv/src/arXiv_src_1903_018.tar",
        "s3://arxiv/src/arXiv_src_1903_019.tar",
        "s3://arxiv/src/arXiv_src_1903_020.tar",
        "s3://arxiv/src/arXiv_src_1903_021.tar",
        "s3://arxiv/src/arXiv_src_1903_022.tar",
        "s3://arxiv/src/arXiv_src_1903_023.tar",
        "s3://arxiv/src/arXiv_src_1903_024.tar",
        "s3://arxiv/src/arXiv_src_1903_025.tar",
        "s3://arxiv/src/arXiv_src_1903_026.tar",
        "s3://arxiv/src/arXiv_src_1903_027.tar",
        "s3://arxiv/src/arXiv_src_1903_028.tar",
        "s3://arxiv/src/arXiv_src_1903_029.tar",
        "s3://arxiv/src/arXiv_src_1903_030.tar",
        "s3://arxiv/src/arXiv_src_1903_031.tar",
        "s3://arxiv/src/arXiv_src_1903_032.tar",
        "s3://arxiv/src/arXiv_src_1903_033.tar",
        "s3://arxiv/src/arXiv_src_1903_034.tar",
        "s3://arxiv/src/arXiv_src_1903_035.tar",
        "s3://arxiv/src/arXiv_src_1903_036.tar",
        "s3://arxiv/src/arXiv_src_1903_037.tar",
        "s3://arxiv/src/arXiv_src_1903_038.tar",
        "s3://arxiv/src/arXiv_src_1904_001.tar",
        "s3://arxiv/src/arXiv_src_1904_002.tar",
        "s3://arxiv/src/arXiv_src_1904_003.tar",
        "s3://arxiv/src/arXiv_src_1904_004.tar",
        "s3://arxiv/src/arXiv_src_1904_005.tar",
        "s3://arxiv/src/arXiv_src_1904_006.tar",
        "s3://arxiv/src/arXiv_src_1904_007.tar",
        "s3://arxiv/src/arXiv_src_1904_008.tar",
        "s3://arxiv/src/arXiv_src_1904_009.tar",
        "s3://arxiv/src/arXiv_src_1904_010.tar",
        "s3://arxiv/src/arXiv_src_1904_011.tar",
        "s3://arxiv/src/arXiv_src_1904_012.tar",
        "s3://arxiv/src/arXiv_src_1904_013.tar",
        "s3://arxiv/src/arXiv_src_1904_014.tar",
        "s3://arxiv/src/arXiv_src_1904_015.tar",
        "s3://arxiv/src/arXiv_src_1904_016.tar",
        "s3://arxiv/src/arXiv_src_1904_017.tar",
        "s3://arxiv/src/arXiv_src_1904_018.tar",
        "s3://arxiv/src/arXiv_src_1904_019.tar",
        "s3://arxiv/src/arXiv_src_1904_020.tar",
        "s3://arxiv/src/arXiv_src_1904_021.tar",
        "s3://arxiv/src/arXiv_src_1904_022.tar",
        "s3://arxiv/src/arXiv_src_1904_023.tar",
        "s3://arxiv/src/arXiv_src_1904_024.tar",
        "s3://arxiv/src/arXiv_src_1904_025.tar",
        "s3://arxiv/src/arXiv_src_1904_026.tar",
        "s3://arxiv/src/arXiv_src_1904_027.tar",
        "s3://arxiv/src/arXiv_src_1904_028.tar",
        "s3://arxiv/src/arXiv_src_1904_029.tar",
        "s3://arxiv/src/arXiv_src_1904_030.tar",
        "s3://arxiv/src/arXiv_src_1904_031.tar",
        "s3://arxiv/src/arXiv_src_1904_032.tar",
        "s3://arxiv/src/arXiv_src_1904_033.tar",
        "s3://arxiv/src/arXiv_src_1904_034.tar",
        "s3://arxiv/src/arXiv_src_1904_035.tar",
        "s3://arxiv/src/arXiv_src_1904_036.tar",
        "s3://arxiv/src/arXiv_src_1904_037.tar",
        "s3://arxiv/src/arXiv_src_1904_038.tar",
        "s3://arxiv/src/arXiv_src_1904_039.tar",
        "s3://arxiv/src/arXiv_src_1904_040.tar",
        "s3://arxiv/src/arXiv_src_1904_041.tar",
        "s3://arxiv/src/arXiv_src_1904_042.tar",
        "s3://arxiv/src/arXiv_src_1905_001.tar",
        "s3://arxiv/src/arXiv_src_1905_002.tar",
        "s3://arxiv/src/arXiv_src_1905_003.tar",
        "s3://arxiv/src/arXiv_src_1905_004.tar",
        "s3://arxiv/src/arXiv_src_1905_005.tar",
        "s3://arxiv/src/arXiv_src_1905_006.tar",
        "s3://arxiv/src/arXiv_src_1905_007.tar",
        "s3://arxiv/src/arXiv_src_1905_008.tar",
        "s3://arxiv/src/arXiv_src_1905_009.tar",
        "s3://arxiv/src/arXiv_src_1905_010.tar",
        "s3://arxiv/src/arXiv_src_1905_011.tar",
        "s3://arxiv/src/arXiv_src_1905_012.tar",
        "s3://arxiv/src/arXiv_src_1905_013.tar",
        "s3://arxiv/src/arXiv_src_1905_014.tar",
        "s3://arxiv/src/arXiv_src_1905_015.tar",
        "s3://arxiv/src/arXiv_src_1905_016.tar",
        "s3://arxiv/src/arXiv_src_1905_017.tar",
        "s3://arxiv/src/arXiv_src_1905_018.tar",
        "s3://arxiv/src/arXiv_src_1905_019.tar",
        "s3://arxiv/src/arXiv_src_1905_020.tar",
        "s3://arxiv/src/arXiv_src_1905_021.tar",
        "s3://arxiv/src/arXiv_src_1905_022.tar",
        "s3://arxiv/src/arXiv_src_1905_023.tar",
        "s3://arxiv/src/arXiv_src_1905_024.tar",
        "s3://arxiv/src/arXiv_src_1905_025.tar",
        "s3://arxiv/src/arXiv_src_1905_026.tar",
        "s3://arxiv/src/arXiv_src_1905_027.tar",
        "s3://arxiv/src/arXiv_src_1905_028.tar",
        "s3://arxiv/src/arXiv_src_1905_029.tar",
        "s3://arxiv/src/arXiv_src_1905_030.tar",
        "s3://arxiv/src/arXiv_src_1905_031.tar",
        "s3://arxiv/src/arXiv_src_1905_032.tar",
        "s3://arxiv/src/arXiv_src_1905_033.tar",
        "s3://arxiv/src/arXiv_src_1905_034.tar",
        "s3://arxiv/src/arXiv_src_1905_035.tar",
        "s3://arxiv/src/arXiv_src_1905_036.tar",
        "s3://arxiv/src/arXiv_src_1905_037.tar",
        "s3://arxiv/src/arXiv_src_1905_038.tar",
        "s3://arxiv/src/arXiv_src_1905_039.tar",
        "s3://arxiv/src/arXiv_src_1905_040.tar",
        "s3://arxiv/src/arXiv_src_1905_041.tar",
        "s3://arxiv/src/arXiv_src_1906_001.tar",
        "s3://arxiv/src/arXiv_src_1906_002.tar",
        "s3://arxiv/src/arXiv_src_1906_003.tar",
        "s3://arxiv/src/arXiv_src_1906_004.tar",
        "s3://arxiv/src/arXiv_src_1906_005.tar",
        "s3://arxiv/src/arXiv_src_1906_006.tar",
        "s3://arxiv/src/arXiv_src_1906_007.tar",
        "s3://arxiv/src/arXiv_src_1906_008.tar",
        "s3://arxiv/src/arXiv_src_1906_009.tar",
        "s3://arxiv/src/arXiv_src_1906_010.tar",
        "s3://arxiv/src/arXiv_src_1906_011.tar",
        "s3://arxiv/src/arXiv_src_1906_012.tar",
        "s3://arxiv/src/arXiv_src_1906_013.tar",
        "s3://arxiv/src/arXiv_src_1906_014.tar",
        "s3://arxiv/src/arXiv_src_1906_015.tar",
        "s3://arxiv/src/arXiv_src_1906_016.tar",
        "s3://arxiv/src/arXiv_src_1906_017.tar",
        "s3://arxiv/src/arXiv_src_1906_018.tar",
        "s3://arxiv/src/arXiv_src_1906_019.tar",
        "s3://arxiv/src/arXiv_src_1906_020.tar",
        "s3://arxiv/src/arXiv_src_1906_021.tar",
        "s3://arxiv/src/arXiv_src_1906_022.tar",
        "s3://arxiv/src/arXiv_src_1906_023.tar",
        "s3://arxiv/src/arXiv_src_1906_024.tar",
        "s3://arxiv/src/arXiv_src_1906_025.tar",
        "s3://arxiv/src/arXiv_src_1906_026.tar",
        "s3://arxiv/src/arXiv_src_1906_027.tar",
        "s3://arxiv/src/arXiv_src_1906_028.tar",
        "s3://arxiv/src/arXiv_src_1906_029.tar",
        "s3://arxiv/src/arXiv_src_1906_030.tar",
        "s3://arxiv/src/arXiv_src_1906_031.tar",
        "s3://arxiv/src/arXiv_src_1906_032.tar",
        "s3://arxiv/src/arXiv_src_1906_033.tar",
        "s3://arxiv/src/arXiv_src_1906_034.tar",
        "s3://arxiv/src/arXiv_src_1906_035.tar",
        "s3://arxiv/src/arXiv_src_1906_036.tar",
        "s3://arxiv/src/arXiv_src_1906_037.tar",
        "s3://arxiv/src/arXiv_src_1906_038.tar",
        "s3://arxiv/src/arXiv_src_1907_001.tar",
        "s3://arxiv/src/arXiv_src_1907_002.tar",
        "s3://arxiv/src/arXiv_src_1907_003.tar",
        "s3://arxiv/src/arXiv_src_1907_004.tar",
        "s3://arxiv/src/arXiv_src_1907_005.tar",
        "s3://arxiv/src/arXiv_src_1907_006.tar",
        "s3://arxiv/src/arXiv_src_1907_007.tar",
        "s3://arxiv/src/arXiv_src_1907_008.tar",
        "s3://arxiv/src/arXiv_src_1907_009.tar",
        "s3://arxiv/src/arXiv_src_1907_010.tar",
        "s3://arxiv/src/arXiv_src_1907_011.tar",
        "s3://arxiv/src/arXiv_src_1907_012.tar",
        "s3://arxiv/src/arXiv_src_1907_013.tar",
        "s3://arxiv/src/arXiv_src_1907_014.tar",
        "s3://arxiv/src/arXiv_src_1907_015.tar",
        "s3://arxiv/src/arXiv_src_1907_016.tar",
        "s3://arxiv/src/arXiv_src_1907_017.tar",
        "s3://arxiv/src/arXiv_src_1907_018.tar",
        "s3://arxiv/src/arXiv_src_1907_019.tar",
        "s3://arxiv/src/arXiv_src_1907_020.tar",
        "s3://arxiv/src/arXiv_src_1907_021.tar",
        "s3://arxiv/src/arXiv_src_1907_022.tar",
        "s3://arxiv/src/arXiv_src_1907_023.tar",
        "s3://arxiv/src/arXiv_src_1907_024.tar",
        "s3://arxiv/src/arXiv_src_1907_025.tar",
        "s3://arxiv/src/arXiv_src_1907_026.tar",
        "s3://arxiv/src/arXiv_src_1907_027.tar",
        "s3://arxiv/src/arXiv_src_1907_028.tar",
        "s3://arxiv/src/arXiv_src_1907_029.tar",
        "s3://arxiv/src/arXiv_src_1907_030.tar",
        "s3://arxiv/src/arXiv_src_1907_031.tar",
        "s3://arxiv/src/arXiv_src_1907_032.tar",
        "s3://arxiv/src/arXiv_src_1907_033.tar",
        "s3://arxiv/src/arXiv_src_1907_034.tar",
        "s3://arxiv/src/arXiv_src_1907_035.tar",
        "s3://arxiv/src/arXiv_src_1907_036.tar",
        "s3://arxiv/src/arXiv_src_1907_037.tar",
        "s3://arxiv/src/arXiv_src_1907_038.tar",
        "s3://arxiv/src/arXiv_src_1907_039.tar",
        "s3://arxiv/src/arXiv_src_1907_040.tar",
        "s3://arxiv/src/arXiv_src_1907_041.tar",
        "s3://arxiv/src/arXiv_src_1907_042.tar",
        "s3://arxiv/src/arXiv_src_1908_001.tar",
        "s3://arxiv/src/arXiv_src_1908_002.tar",
        "s3://arxiv/src/arXiv_src_1908_003.tar",
        "s3://arxiv/src/arXiv_src_1908_004.tar",
        "s3://arxiv/src/arXiv_src_1908_005.tar",
        "s3://arxiv/src/arXiv_src_1908_006.tar",
        "s3://arxiv/src/arXiv_src_1908_007.tar",
        "s3://arxiv/src/arXiv_src_1908_008.tar",
        "s3://arxiv/src/arXiv_src_1908_009.tar",
        "s3://arxiv/src/arXiv_src_1908_010.tar",
        "s3://arxiv/src/arXiv_src_1908_011.tar",
        "s3://arxiv/src/arXiv_src_1908_012.tar",
        "s3://arxiv/src/arXiv_src_1908_013.tar",
        "s3://arxiv/src/arXiv_src_1908_014.tar",
        "s3://arxiv/src/arXiv_src_1908_015.tar",
        "s3://arxiv/src/arXiv_src_1908_016.tar",
        "s3://arxiv/src/arXiv_src_1908_017.tar",
        "s3://arxiv/src/arXiv_src_1908_018.tar",
        "s3://arxiv/src/arXiv_src_1908_019.tar",
        "s3://arxiv/src/arXiv_src_1908_020.tar",
        "s3://arxiv/src/arXiv_src_1908_021.tar",
        "s3://arxiv/src/arXiv_src_1908_022.tar",
        "s3://arxiv/src/arXiv_src_1908_023.tar",
        "s3://arxiv/src/arXiv_src_1908_024.tar",
        "s3://arxiv/src/arXiv_src_1908_025.tar",
        "s3://arxiv/src/arXiv_src_1908_026.tar",
        "s3://arxiv/src/arXiv_src_1908_027.tar",
        "s3://arxiv/src/arXiv_src_1908_028.tar",
        "s3://arxiv/src/arXiv_src_1908_029.tar",
        "s3://arxiv/src/arXiv_src_1908_030.tar",
        "s3://arxiv/src/arXiv_src_1908_031.tar",
        "s3://arxiv/src/arXiv_src_1908_032.tar",
        "s3://arxiv/src/arXiv_src_1908_033.tar",
        "s3://arxiv/src/arXiv_src_1908_034.tar",
        "s3://arxiv/src/arXiv_src_1908_035.tar",
        "s3://arxiv/src/arXiv_src_1908_036.tar",
        "s3://arxiv/src/arXiv_src_1908_037.tar",
        "s3://arxiv/src/arXiv_src_1908_038.tar",
        "s3://arxiv/src/arXiv_src_1908_039.tar",
        "s3://arxiv/src/arXiv_src_1909_001.tar",
        "s3://arxiv/src/arXiv_src_1909_002.tar",
        "s3://arxiv/src/arXiv_src_1909_003.tar",
        "s3://arxiv/src/arXiv_src_1909_004.tar",
        "s3://arxiv/src/arXiv_src_1909_005.tar",
        "s3://arxiv/src/arXiv_src_1909_006.tar",
        "s3://arxiv/src/arXiv_src_1909_007.tar",
        "s3://arxiv/src/arXiv_src_1909_008.tar",
        "s3://arxiv/src/arXiv_src_1909_009.tar",
        "s3://arxiv/src/arXiv_src_1909_010.tar",
        "s3://arxiv/src/arXiv_src_1909_011.tar",
        "s3://arxiv/src/arXiv_src_1909_012.tar",
        "s3://arxiv/src/arXiv_src_1909_013.tar",
        "s3://arxiv/src/arXiv_src_1909_014.tar",
        "s3://arxiv/src/arXiv_src_1909_015.tar",
        "s3://arxiv/src/arXiv_src_1909_016.tar",
        "s3://arxiv/src/arXiv_src_1909_017.tar",
        "s3://arxiv/src/arXiv_src_1909_018.tar",
        "s3://arxiv/src/arXiv_src_1909_019.tar",
        "s3://arxiv/src/arXiv_src_1909_020.tar",
        "s3://arxiv/src/arXiv_src_1909_021.tar",
        "s3://arxiv/src/arXiv_src_1909_022.tar",
        "s3://arxiv/src/arXiv_src_1909_023.tar",
        "s3://arxiv/src/arXiv_src_1909_024.tar",
        "s3://arxiv/src/arXiv_src_1909_025.tar",
        "s3://arxiv/src/arXiv_src_1909_026.tar",
        "s3://arxiv/src/arXiv_src_1909_027.tar",
        "s3://arxiv/src/arXiv_src_1909_028.tar",
        "s3://arxiv/src/arXiv_src_1909_029.tar",
        "s3://arxiv/src/arXiv_src_1909_030.tar",
        "s3://arxiv/src/arXiv_src_1909_031.tar",
        "s3://arxiv/src/arXiv_src_1909_032.tar",
        "s3://arxiv/src/arXiv_src_1909_033.tar",
        "s3://arxiv/src/arXiv_src_1909_034.tar",
        "s3://arxiv/src/arXiv_src_1909_035.tar",
        "s3://arxiv/src/arXiv_src_1909_036.tar",
        "s3://arxiv/src/arXiv_src_1909_037.tar",
        "s3://arxiv/src/arXiv_src_1909_038.tar",
        "s3://arxiv/src/arXiv_src_1909_039.tar",
        "s3://arxiv/src/arXiv_src_1909_040.tar",
        "s3://arxiv/src/arXiv_src_1909_041.tar",
        "s3://arxiv/src/arXiv_src_1909_042.tar",
        "s3://arxiv/src/arXiv_src_1909_043.tar",
        "s3://arxiv/src/arXiv_src_1909_044.tar",
        "s3://arxiv/src/arXiv_src_1910_001.tar",
        "s3://arxiv/src/arXiv_src_1910_002.tar",
        "s3://arxiv/src/arXiv_src_1910_003.tar",
        "s3://arxiv/src/arXiv_src_1910_004.tar",
        "s3://arxiv/src/arXiv_src_1910_005.tar",
        "s3://arxiv/src/arXiv_src_1910_006.tar",
        "s3://arxiv/src/arXiv_src_1910_007.tar",
        "s3://arxiv/src/arXiv_src_1910_008.tar",
        "s3://arxiv/src/arXiv_src_1910_009.tar",
        "s3://arxiv/src/arXiv_src_1910_010.tar",
        "s3://arxiv/src/arXiv_src_1910_011.tar",
        "s3://arxiv/src/arXiv_src_1910_012.tar",
        "s3://arxiv/src/arXiv_src_1910_013.tar",
        "s3://arxiv/src/arXiv_src_1910_014.tar",
        "s3://arxiv/src/arXiv_src_1910_015.tar",
        "s3://arxiv/src/arXiv_src_1910_016.tar",
        "s3://arxiv/src/arXiv_src_1910_017.tar",
        "s3://arxiv/src/arXiv_src_1910_018.tar",
        "s3://arxiv/src/arXiv_src_1910_019.tar",
        "s3://arxiv/src/arXiv_src_1910_020.tar",
        "s3://arxiv/src/arXiv_src_1910_021.tar",
        "s3://arxiv/src/arXiv_src_1910_022.tar",
        "s3://arxiv/src/arXiv_src_1910_023.tar",
        "s3://arxiv/src/arXiv_src_1910_024.tar",
        "s3://arxiv/src/arXiv_src_1910_025.tar",
        "s3://arxiv/src/arXiv_src_1910_026.tar",
        "s3://arxiv/src/arXiv_src_1910_027.tar",
        "s3://arxiv/src/arXiv_src_1910_028.tar",
        "s3://arxiv/src/arXiv_src_1910_029.tar",
        "s3://arxiv/src/arXiv_src_1910_030.tar",
        "s3://arxiv/src/arXiv_src_1910_031.tar",
        "s3://arxiv/src/arXiv_src_1910_032.tar",
        "s3://arxiv/src/arXiv_src_1910_033.tar",
        "s3://arxiv/src/arXiv_src_1910_034.tar",
        "s3://arxiv/src/arXiv_src_1910_035.tar",
        "s3://arxiv/src/arXiv_src_1910_036.tar",
        "s3://arxiv/src/arXiv_src_1910_037.tar",
        "s3://arxiv/src/arXiv_src_1910_038.tar",
        "s3://arxiv/src/arXiv_src_1910_039.tar",
        "s3://arxiv/src/arXiv_src_1910_040.tar",
        "s3://arxiv/src/arXiv_src_1910_041.tar",
        "s3://arxiv/src/arXiv_src_1910_042.tar",
        "s3://arxiv/src/arXiv_src_1910_043.tar",
        "s3://arxiv/src/arXiv_src_1910_044.tar",
        "s3://arxiv/src/arXiv_src_1910_045.tar",
        "s3://arxiv/src/arXiv_src_1910_046.tar",
        "s3://arxiv/src/arXiv_src_1910_047.tar",
        "s3://arxiv/src/arXiv_src_1911_001.tar",
        "s3://arxiv/src/arXiv_src_1911_002.tar",
        "s3://arxiv/src/arXiv_src_1911_003.tar",
        "s3://arxiv/src/arXiv_src_1911_004.tar",
        "s3://arxiv/src/arXiv_src_1911_005.tar",
        "s3://arxiv/src/arXiv_src_1911_006.tar",
        "s3://arxiv/src/arXiv_src_1911_007.tar",
        "s3://arxiv/src/arXiv_src_1911_008.tar",
        "s3://arxiv/src/arXiv_src_1911_009.tar",
        "s3://arxiv/src/arXiv_src_1911_010.tar",
        "s3://arxiv/src/arXiv_src_1911_011.tar",
        "s3://arxiv/src/arXiv_src_1911_012.tar",
        "s3://arxiv/src/arXiv_src_1911_013.tar",
        "s3://arxiv/src/arXiv_src_1911_014.tar",
        "s3://arxiv/src/arXiv_src_1911_015.tar",
        "s3://arxiv/src/arXiv_src_1911_016.tar",
        "s3://arxiv/src/arXiv_src_1911_017.tar",
        "s3://arxiv/src/arXiv_src_1911_018.tar",
        "s3://arxiv/src/arXiv_src_1911_019.tar",
        "s3://arxiv/src/arXiv_src_1911_020.tar",
        "s3://arxiv/src/arXiv_src_1911_021.tar",
        "s3://arxiv/src/arXiv_src_1911_022.tar",
        "s3://arxiv/src/arXiv_src_1911_023.tar",
        "s3://arxiv/src/arXiv_src_1911_024.tar",
        "s3://arxiv/src/arXiv_src_1911_025.tar",
        "s3://arxiv/src/arXiv_src_1911_026.tar",
        "s3://arxiv/src/arXiv_src_1911_027.tar",
        "s3://arxiv/src/arXiv_src_1911_028.tar",
        "s3://arxiv/src/arXiv_src_1911_029.tar",
        "s3://arxiv/src/arXiv_src_1911_030.tar",
        "s3://arxiv/src/arXiv_src_1911_031.tar",
        "s3://arxiv/src/arXiv_src_1911_032.tar",
        "s3://arxiv/src/arXiv_src_1911_033.tar",
        "s3://arxiv/src/arXiv_src_1911_034.tar",
        "s3://arxiv/src/arXiv_src_1911_035.tar",
        "s3://arxiv/src/arXiv_src_1911_036.tar",
        "s3://arxiv/src/arXiv_src_1911_037.tar",
        "s3://arxiv/src/arXiv_src_1911_038.tar",
        "s3://arxiv/src/arXiv_src_1911_039.tar",
        "s3://arxiv/src/arXiv_src_1911_040.tar",
        "s3://arxiv/src/arXiv_src_1911_041.tar",
        "s3://arxiv/src/arXiv_src_1911_042.tar",
        "s3://arxiv/src/arXiv_src_1911_043.tar",
        "s3://arxiv/src/arXiv_src_1912_001.tar",
        "s3://arxiv/src/arXiv_src_1912_002.tar",
        "s3://arxiv/src/arXiv_src_1912_003.tar",
        "s3://arxiv/src/arXiv_src_1912_004.tar",
        "s3://arxiv/src/arXiv_src_1912_005.tar",
        "s3://arxiv/src/arXiv_src_1912_006.tar",
        "s3://arxiv/src/arXiv_src_1912_007.tar",
        "s3://arxiv/src/arXiv_src_1912_008.tar",
        "s3://arxiv/src/arXiv_src_1912_009.tar",
        "s3://arxiv/src/arXiv_src_1912_010.tar",
        "s3://arxiv/src/arXiv_src_1912_011.tar",
        "s3://arxiv/src/arXiv_src_1912_012.tar",
        "s3://arxiv/src/arXiv_src_1912_013.tar",
        "s3://arxiv/src/arXiv_src_1912_014.tar",
        "s3://arxiv/src/arXiv_src_1912_015.tar",
        "s3://arxiv/src/arXiv_src_1912_016.tar",
        "s3://arxiv/src/arXiv_src_1912_017.tar",
        "s3://arxiv/src/arXiv_src_1912_018.tar",
        "s3://arxiv/src/arXiv_src_1912_019.tar",
        "s3://arxiv/src/arXiv_src_1912_020.tar",
        "s3://arxiv/src/arXiv_src_1912_021.tar",
        "s3://arxiv/src/arXiv_src_1912_022.tar",
        "s3://arxiv/src/arXiv_src_1912_023.tar",
        "s3://arxiv/src/arXiv_src_1912_024.tar",
        "s3://arxiv/src/arXiv_src_1912_025.tar",
        "s3://arxiv/src/arXiv_src_1912_026.tar",
        "s3://arxiv/src/arXiv_src_1912_027.tar",
        "s3://arxiv/src/arXiv_src_1912_028.tar",
        "s3://arxiv/src/arXiv_src_1912_029.tar",
        "s3://arxiv/src/arXiv_src_1912_030.tar",
        "s3://arxiv/src/arXiv_src_1912_031.tar",
        "s3://arxiv/src/arXiv_src_1912_032.tar",
        "s3://arxiv/src/arXiv_src_1912_033.tar",
        "s3://arxiv/src/arXiv_src_1912_034.tar",
        "s3://arxiv/src/arXiv_src_1912_035.tar",
        "s3://arxiv/src/arXiv_src_1912_036.tar",
        "s3://arxiv/src/arXiv_src_1912_037.tar",
        "s3://arxiv/src/arXiv_src_1912_038.tar",
        "s3://arxiv/src/arXiv_src_1912_039.tar",
        "s3://arxiv/src/arXiv_src_1912_040.tar",
        "s3://arxiv/src/arXiv_src_1912_041.tar",
        "s3://arxiv/src/arXiv_src_1912_042.tar",
        "s3://arxiv/src/arXiv_src_1912_043.tar",
        "s3://arxiv/src/arXiv_src_1912_044.tar",
        "s3://arxiv/src/arXiv_src_9107_001.tar",
        "s3://arxiv/src/arXiv_src_9108_001.tar",
        "s3://arxiv/src/arXiv_src_9109_001.tar",
        "s3://arxiv/src/arXiv_src_9110_001.tar",
        "s3://arxiv/src/arXiv_src_9111_001.tar",
        "s3://arxiv/src/arXiv_src_9112_001.tar",
        "s3://arxiv/src/arXiv_src_9201_001.tar",
        "s3://arxiv/src/arXiv_src_9202_001.tar",
        "s3://arxiv/src/arXiv_src_9203_001.tar",
        "s3://arxiv/src/arXiv_src_9204_001.tar",
        "s3://arxiv/src/arXiv_src_9205_001.tar",
        "s3://arxiv/src/arXiv_src_9206_001.tar",
        "s3://arxiv/src/arXiv_src_9207_001.tar",
        "s3://arxiv/src/arXiv_src_9208_001.tar",
        "s3://arxiv/src/arXiv_src_9209_001.tar",
        "s3://arxiv/src/arXiv_src_9210_001.tar",
        "s3://arxiv/src/arXiv_src_9211_001.tar",
        "s3://arxiv/src/arXiv_src_9212_001.tar",
        "s3://arxiv/src/arXiv_src_9301_001.tar",
        "s3://arxiv/src/arXiv_src_9302_001.tar",
        "s3://arxiv/src/arXiv_src_9303_001.tar",
        "s3://arxiv/src/arXiv_src_9304_001.tar",
        "s3://arxiv/src/arXiv_src_9305_001.tar",
        "s3://arxiv/src/arXiv_src_9306_001.tar",
        "s3://arxiv/src/arXiv_src_9307_001.tar",
        "s3://arxiv/src/arXiv_src_9308_001.tar",
        "s3://arxiv/src/arXiv_src_9309_001.tar",
        "s3://arxiv/src/arXiv_src_9310_001.tar",
        "s3://arxiv/src/arXiv_src_9311_001.tar",
        "s3://arxiv/src/arXiv_src_9312_001.tar",
        "s3://arxiv/src/arXiv_src_9401_001.tar",
        "s3://arxiv/src/arXiv_src_9402_001.tar",
        "s3://arxiv/src/arXiv_src_9403_001.tar",
        "s3://arxiv/src/arXiv_src_9404_001.tar",
        "s3://arxiv/src/arXiv_src_9405_001.tar",
        "s3://arxiv/src/arXiv_src_9406_001.tar",
        "s3://arxiv/src/arXiv_src_9407_001.tar",
        "s3://arxiv/src/arXiv_src_9408_001.tar",
        "s3://arxiv/src/arXiv_src_9409_001.tar",
        "s3://arxiv/src/arXiv_src_9410_001.tar",
        "s3://arxiv/src/arXiv_src_9411_001.tar",
        "s3://arxiv/src/arXiv_src_9412_001.tar",
        "s3://arxiv/src/arXiv_src_9501_001.tar",
        "s3://arxiv/src/arXiv_src_9502_001.tar",
        "s3://arxiv/src/arXiv_src_9503_001.tar",
        "s3://arxiv/src/arXiv_src_9504_001.tar",
        "s3://arxiv/src/arXiv_src_9505_001.tar",
        "s3://arxiv/src/arXiv_src_9506_001.tar",
        "s3://arxiv/src/arXiv_src_9507_001.tar",
        "s3://arxiv/src/arXiv_src_9508_001.tar",
        "s3://arxiv/src/arXiv_src_9509_001.tar",
        "s3://arxiv/src/arXiv_src_9510_001.tar",
        "s3://arxiv/src/arXiv_src_9511_001.tar",
        "s3://arxiv/src/arXiv_src_9512_001.tar",
        "s3://arxiv/src/arXiv_src_9601_001.tar",
        "s3://arxiv/src/arXiv_src_9602_001.tar",
        "s3://arxiv/src/arXiv_src_9603_001.tar",
        "s3://arxiv/src/arXiv_src_9604_001.tar",
        "s3://arxiv/src/arXiv_src_9605_001.tar",
        "s3://arxiv/src/arXiv_src_9606_001.tar",
        "s3://arxiv/src/arXiv_src_9607_001.tar",
        "s3://arxiv/src/arXiv_src_9608_001.tar",
        "s3://arxiv/src/arXiv_src_9609_001.tar",
        "s3://arxiv/src/arXiv_src_9610_001.tar",
        "s3://arxiv/src/arXiv_src_9611_001.tar",
        "s3://arxiv/src/arXiv_src_9612_001.tar",
        "s3://arxiv/src/arXiv_src_9701_001.tar",
        "s3://arxiv/src/arXiv_src_9702_001.tar",
        "s3://arxiv/src/arXiv_src_9703_001.tar",
        "s3://arxiv/src/arXiv_src_9704_001.tar",
        "s3://arxiv/src/arXiv_src_9705_001.tar",
        "s3://arxiv/src/arXiv_src_9706_001.tar",
        "s3://arxiv/src/arXiv_src_9707_001.tar",
        "s3://arxiv/src/arXiv_src_9708_001.tar",
        "s3://arxiv/src/arXiv_src_9709_001.tar",
        "s3://arxiv/src/arXiv_src_9710_001.tar",
        "s3://arxiv/src/arXiv_src_9711_001.tar",
        "s3://arxiv/src/arXiv_src_9712_001.tar",
        "s3://arxiv/src/arXiv_src_9801_001.tar",
        "s3://arxiv/src/arXiv_src_9802_001.tar",
        "s3://arxiv/src/arXiv_src_9803_001.tar",
        "s3://arxiv/src/arXiv_src_9804_001.tar",
        "s3://arxiv/src/arXiv_src_9805_001.tar",
        "s3://arxiv/src/arXiv_src_9806_001.tar",
        "s3://arxiv/src/arXiv_src_9807_001.tar",
        "s3://arxiv/src/arXiv_src_9808_001.tar",
        "s3://arxiv/src/arXiv_src_9809_001.tar",
        "s3://arxiv/src/arXiv_src_9810_001.tar",
        "s3://arxiv/src/arXiv_src_9811_001.tar",
        "s3://arxiv/src/arXiv_src_9812_001.tar",
        "s3://arxiv/src/arXiv_src_9901_001.tar",
        "s3://arxiv/src/arXiv_src_9902_001.tar",
        "s3://arxiv/src/arXiv_src_9903_001.tar",
        "s3://arxiv/src/arXiv_src_9904_001.tar",
        "s3://arxiv/src/arXiv_src_9905_001.tar",
        "s3://arxiv/src/arXiv_src_9906_001.tar",
        "s3://arxiv/src/arXiv_src_9907_001.tar",
        "s3://arxiv/src/arXiv_src_9908_001.tar",
        "s3://arxiv/src/arXiv_src_9909_001.tar",
        "s3://arxiv/src/arXiv_src_9910_001.tar",
        "s3://arxiv/src/arXiv_src_9911_001.tar",
        "s3://arxiv/src/arXiv_src_9912_001.tar"
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
