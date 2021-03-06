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
import json

import numpy as np
from skimage import measure
from PIL import Image
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
        color_page_im = imageio.imread(color_page)
        black_page_im = imageio.imread(black_page)
        assert color_page_im.shape == black_page_im.shape
        diff_page = figure_utils.im_diff(color_page_im, black_page_im)
        diff_name = result_dir + 'diff-' + os.path.basename(black_page)
        imageio.imwrite(diff_name, diff_page)
        # imageio.imwrite(diff_name, )
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
        page_image = imageio.imread(page_image_name)
        diff_im = imageio.imread(diff)
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
        tarname: str, extract_dir: str, n_attempts: int = 100, cache_dir: str = settings.ARXIV_DATA_CACHE_DIR,
        delete_tar_after_extracting: bool = False
) -> None:
    print('.', end='', flush=True)
    logging.info('Downloading %s' % tarname)
    for attempt in range(n_attempts):
        try:
            cached_file = file_util.cache_file_2(tarname, cache_dir=cache_dir)
            break
        except FileNotFoundError:
            if attempt == n_attempts - 1:
                raise
            logging.exception('Download failed, retrying')
            time.sleep(10)
    logging.info("Proceeding to extract tar file: {}".format(cached_file))
    file_util.extract_tarfile(cached_file, extract_dir)
    if delete_tar_after_extracting:
        os.remove(cached_file)


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
    tarnames = json.load(open(settings.FILE_LIST))
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
        paper_tarnames = glob.glob(os.path.join(tmpdir, '*/*.gz'))
        print(datetime.datetime.now())
        print(
            'Processing %d papers in group %s' %
            (len(paper_tarnames), str(group_key))
        )
        with multiprocessing.Pool(processes=round(settings.PROCESS_PAPER_TAR_THREAD_COUNT)
                                  ) as p:
            p.map(process_paper_tar, paper_tarnames)


if __name__ == "__main__":
    if settings.IN_DOCKER:
        logging.basicConfig(filename='/work/host-output/logger_arxiv.log', level=logging.INFO)
    else:
        logging.basicConfig(filename='logger_arxiv.log', level=logging.INFO)
    run_on_all()
    print('All done')
