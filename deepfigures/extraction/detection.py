"""Functions for detecting and extracting figures."""

import os
import json
from deepfigures.extraction.datamodels import BoxClass
from joblib import Parallel, delayed
import multiprocessing

from typing import List, Tuple, Iterable

import cv2  # Need to import OpenCV before tensorflow to avoid import error
import imageio
import numpy as np

from deepfigures.extraction import (
    tensorbox_fourchannel,
    pdffigures_wrapper,
    figure_utils)
from deepfigures import settings
from deepfigures.extraction.datamodels import (
    BoxClass,
    Figure,
    PdfDetectionResult,
    CaptionOnly)
from deepfigures import settings
from deepfigures.utils import (
    file_util,
    settings_utils)
from deepfigures.utils import misc

PAD_FACTOR = 0.02
TENSORBOX_MODEL = settings.TENSORBOX_MODEL

# Holds a cached instantiation of TensorboxCaptionmaskDetector.
_detector = None


def get_detector(detector_args=TENSORBOX_MODEL) -> tensorbox_fourchannel.TensorboxCaptionmaskDetector:
    """
    Get TensorboxCaptionmaskDetector instance, initializing it on the first call.
    """
    global _detector
    if not _detector:
        _detector = tensorbox_fourchannel.TensorboxCaptionmaskDetector(
            **detector_args)
    return _detector


def extract_figures_json(
        pdf_path,
        page_image_paths,
        pdffigures_output,
        output_directory):
    """Extract information about figures to JSON and save to disk.

    :param str pdf_path: path to the PDF from which to extract
      figures.

    :returns: path to the JSON file containing the detection results.
    """
    page_images_array = np.array([
        imageio.imread(page_image_path)
        for page_image_path in page_image_paths
    ])
    detector = get_detector()
    figure_boxes_by_page = detector.get_detections(
        page_images_array)
    pdffigures_captions = pdffigures_wrapper.get_captions(
        pdffigures_output=pdffigures_output,
        target_dpi=settings.DEFAULT_INFERENCE_DPI)
    figures_by_page = []
    for page_num in range(len(page_image_paths)):
        figure_boxes = figure_boxes_by_page[page_num]
        pf_page_captions = [
            caption
            for caption in pdffigures_captions
            if caption.page == page_num
        ]
        caption_boxes = [
            caption.caption_boundary
            for caption in pf_page_captions
        ]
        figure_indices, caption_indices = figure_utils.pair_boxes(
            figure_boxes, caption_boxes)
        page_image = page_images_array[page_num]
        pad_pixels = PAD_FACTOR * min(page_image.shape[:2])
        for (figure_idx, caption_idx) in zip(figure_indices, caption_indices):
            figures_by_page.append(
                Figure(
                    figure_boundary=figure_boxes[figure_idx].expand_box(
                        pad_pixels).crop_to_page(
                        page_image.shape).crop_whitespace_edges(
                        page_image),
                    caption_boundary=caption_boxes[caption_idx],
                    caption_text=pf_page_captions[caption_idx].caption_text,
                    name=pf_page_captions[caption_idx].name,
                    figure_type=pf_page_captions[caption_idx].figure_type,
                    page=page_num))
    pdf_detection_result = PdfDetectionResult(
        pdf=pdf_path,
        figures=figures_by_page,
        dpi=settings.DEFAULT_INFERENCE_DPI,
        raw_detected_boxes=figure_boxes_by_page,
        raw_pdffigures_output=pdffigures_output)

    output_path = os.path.join(
        output_directory,
        os.path.basename(pdf_path)[:-4] + 'deepfigures-results.json')
    file_util.write_json_atomic(
        output_path,
        pdf_detection_result.to_dict(),
        indent=2,
        sort_keys=True)
    return output_path


# def read_image(image_path: str) -> np.ndarray:
#     return imageio.imread()
#     pass
#
#
# def read_images_and_add_to_queue(queue: multiprocessing.Queue, image_paths: List[str]) -> None:
#
#     queue.put(imageio.imread(im))
#     pass


def run_detection_on_coco_dataset(dataset_dir: str, images_sub_dir: str, model_save_dir: str, iteration: int,
                                  output_json_file_name: str, batch_size: int = 100):
    num_cores = multiprocessing.cpu_count()
    detector_args = {
        'save_dir': model_save_dir,
        'iteration': iteration
    }
    detector = get_detector(detector_args=detector_args)
    annos = json.load(open(os.path.join(dataset_dir, 'figure_boundaries.json')))
    anno_batches = [annos[i:i + batch_size] for i in range(0, len(annos), batch_size)]
    processed_annos = []
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    done_counter = 0
    for anno_batch in anno_batches:
        _image_paths = [os.path.join(dataset_dir, images_sub_dir, anno['image_path']) for anno in anno_batch]
        np_image_list = pool.map(imageio.imread, _image_paths)
        _figure_boxes_by_page = detector.get_detections(np_image_list)
        assert len(_figure_boxes_by_page) == len(np_image_list)
        for idx, anno in enumerate(anno_batch):
            processed_anno = anno
            processed_anno['hidden_set_rects'] = [{'x1': box.x1, 'y1': box.y1, 'x2': box.x2, 'y2': box.y2, } for box
                                                  in _figure_boxes_by_page[idx]]
            processed_annos.append(processed_anno)
        json.dump(processed_annos, open(os.path.join(model_save_dir, output_json_file_name), mode='w'), indent=2)
        done_counter = done_counter + batch_size
        print("Finished processing: {}".format(done_counter))


def evaluate_dataset_on_weights(hidden_set_dir: str, hidden_set_images_subdir: str, save_dir: str, iteration: int):
    detector_args = {
        'save_dir': save_dir,
        'iteration': iteration
    }
    detector = get_detector(detector_args=detector_args)
    annos = json.load(open(os.path.join(hidden_set_dir, 'figure_boundaries.json')))

    for anno in annos:
        image_np = imageio.imread(os.path.join(hidden_set_dir, hidden_set_images_subdir, anno['image_path']))
        pred_boxes = detector.get_detections([image_np])[0]
        true_boxes = [BoxClass(x1=rect['x1'], y1=rect['y1'], x2=rect['x2'], y2=rect['y2']) for rect in anno['rects']]
        (pred_indices, true_indices) = figure_utils.pair_boxes(pred_boxes, true_boxes)
        print("Pred boxes: ", pred_boxes)
        print("True boxes: ", true_boxes)
        print("Pred indices: ", pred_indices)
        print("True indices: ", true_indices)
        a = 0
