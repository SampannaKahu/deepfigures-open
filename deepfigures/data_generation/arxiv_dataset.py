import torch
from torch.utils.data.dataloader import DataLoader

import os
import cv2
import glob
import queue
import shutil
import random
import logging
import numpy as np

from multiprocessing import Manager

from deepfigures import settings
from deepfigures.data_generation import arxiv_pipeline
from deepfigures.data_generation import utils
from deepfigures.data_generation.paper_tar_processor import PaperTarProcessor

FILE_NAME = 'file_name'
PAPER_TAR_NAMES = 'paper_tar_names'
FIGURE_JSONS = 'figure_jsons'


def separate_figure_boundary(figure_boundary):
    if len(figure_boundary['rects']) == 0 or not figure_boundary['rects']:
        return [figure_boundary]
    else:
        result = []
        for rect in figure_boundary['rects']:
            separated_fig_boundary = {
                'image_path': figure_boundary['image_path'],
                'rects': [rect]
            }
            result.append(separated_fig_boundary)
        return result


def separate_figure_boundaries(figure_boundaries):
    result = []
    for figure_boundary in figure_boundaries:
        result = result + separate_figure_boundary(figure_boundary)
    return result


class ArxivDataSet(torch.utils.data.dataset.IterableDataset):
    """
    This class represents the ArXiv dataset of scholarly papers available here: https://arxiv.org/help/bulk_data_s3
    This class inherits from the class torch.utils.data.dataset.IterableDataset. An iterable dataset is especially
    useful when processing streams because the length of the stream cannot be known util the stream completes.
    This is also true for the ArXiv dataset because the total number of data points in the entire dataset cannot be
    estimated easily because:
        - The dataset is stored in zipped files of not more than 500 MB. (Approx. 2366 total zip files as of March 2020)
        - Each zipped file contains a variable number of scholarly papers.
        - For certain use-cases, not all the scholarly papers in each of these zip files would be a valid data point of
        the dataset.
    Therefore, this class supports multi-processed reading of data-points from a single instance of the arxiv dataset
    on the disk. This is implemented by the feature of the PyTorch library to support multiple workers. Details about
    the multi-worker support from pyTorch can be found here: https://pytorch.org/docs/stable/data.html#iterable-style-datasets.
    Further details about the implementation are provided in the documentation of each method of this class.
    """

    def __init__(self, list_of_files=None, shuffle_input=True, work_dir_prefix: str = settings.HOSTNAME,
                 arxiv_tmp_dir: str = settings.ARXIV_DATA_TMP_DIR,
                 arxiv_cache_dir: str = settings.ARXIV_DATA_CACHE_DIR,
                 arxiv_data_output_dir: str = settings.ARXIV_DATA_OUTPUT_DIR,
                 get_raw_image: bool = True,
                 delete_tar_after_extracting: bool = False,
                 augment_typewriter_font=True,
                 augment_line_spacing_1_5=True,
                 image_augmentation_transform_sequence=settings.seq,
                 ignore_pages_with_no_figures: bool = False) -> None:
        """
        This class initializes the queue and the contexts for each worker.
        Irrespective of the number of workers in the DataLoader, this constructor will be
        called only once.
        Hence, there will be only one instance of the class variables of this class shared across all workers.
        Further, each worker runs in it's separate process (not a separate thread).
        More details on process vs thread are here: https://www.slashroot.in/difference-between-process-and-thread-linux
        Thus, to ensure that we share the class variables across different workers in a safe manner, we make use of the
        multiprocessing.Manager.Queue class.
        References:
            - https://docs.python.org/3/library/multiprocessing.html#multiprocessing.sharedctypes.multiprocessing.Manager
            - https://docs.python.org/3/library/multiprocessing.html#multiprocessing.managers.SyncManager
            - https://docs.python.org/3/library/multiprocessing.html#multiprocessing.managers.SyncManager.Queue
        The worker context should be available for only this process and should not be shared across different
        processes. Hence, we do not wrap it with the multiprocessing.Manager class.

        :param list_of_files: is the list of the zipped files that we want to read.
        :param shuffle_input: if True, will randomly shuffle the input.
        """
        super().__init__()
        self.work_dir_prefix = work_dir_prefix
        self.arxiv_tmp_dir = arxiv_tmp_dir
        self.arxiv_cache_dir = arxiv_cache_dir
        self.arxiv_data_output_dir = arxiv_data_output_dir
        self.get_raw_image = get_raw_image
        self.delete_tar_after_extracting = delete_tar_after_extracting
        self.augment_typewriter_font = augment_typewriter_font
        self.augment_line_spacing_1_5 = augment_line_spacing_1_5
        self.image_augmentation_transform_sequence = image_augmentation_transform_sequence
        self.ignore_pages_with_no_figures = ignore_pages_with_no_figures

        if not list_of_files:
            list_of_files = []

        if shuffle_input:
            random.shuffle(list_of_files)  # in-place shuffling.

        q = Manager().Queue(maxsize=len(list_of_files))  # The Manager() class ensures safety between worker processes.

        for file in list_of_files:
            q.put(file)

        self.worker_id_to_context_map = {}
        self.q = q

    def _populate_worker_context(self, worker_id: int, file_name: str, delete_tar_after_extracting: bool) -> None:
        """
        Cleans up the tmp directory for this worker.
        Downloads and unzips the file in the tmp directory.
        Find the paths to all the paper tars.
        Populates the worker context with the filename and the paper tar names for processing.

        :param worker_id: the id of the current worker.
        :param file_name: the name of the zipped file that this worker will process next.
        :return: None.
        """
        worker_tmpdir = self.arxiv_tmp_dir + '/' + self.work_dir_prefix + '_' + str(worker_id) + '/'
        if os.path.exists(worker_tmpdir):
            shutil.rmtree(worker_tmpdir)
        os.makedirs(worker_tmpdir)
        arxiv_pipeline.download_and_extract_tar(file_name, extract_dir=worker_tmpdir, cache_dir=self.arxiv_cache_dir,
                                                delete_tar_after_extracting=delete_tar_after_extracting)
        paper_tarnames = glob.glob(os.path.join(worker_tmpdir, '*/*.gz'))
        self.worker_id_to_context_map[worker_id] = {
            FILE_NAME: file_name,
            PAPER_TAR_NAMES: paper_tarnames,
            FIGURE_JSONS: []
        }

    def __iter__(self):
        """
        This method returns an iterator for iterating over eac data-point in the dataset.
        We have implemented the __next__() method for this class. This makes this class an iterator. Hence, simply
        returning the current instance will suffice.

        :return: an iterator for iterating over each data-point in the dataset. (i.e. self in this case.)
        """
        return self

    def __next__(self):
        """
        Any class in Python which implements the  __next__() method can serve as an iterator.
        In the the __iter__() function of this class, we return an instance of self.
        Therefore, to obtain each data-point of the dataset, this method is called.

        Specifically, this method does the following:
            - Obtains the worker info to obtain the worker_id.
            - Returns the next available figure_json from this worker's worker context.
            - If no more figure_jsons are available in this workers context, try to generate more from the next paper
            tar.
            - If no more paper_tar_names are available in this worker's context, try to get the next file from the
            shared queue and unzip it to obtain the paper_tar_names.
            - If the shared queue is empty, stop the iteration.

        :return: the next figure_json from the worker's context.
        """
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id

        # If no more bounding boxes are available, try to generate them.
        if not self.worker_id_to_context_map.get(worker_id) or not self.worker_id_to_context_map[worker_id][
            FIGURE_JSONS]:

            # If paper tars are available, try to process each tar.
            figure_boundaries = []
            while not figure_boundaries:
                # If no more paper tars are available, try to generate them.
                if not self.worker_id_to_context_map.get(worker_id) or not self.worker_id_to_context_map[worker_id][
                    PAPER_TAR_NAMES]:
                    try:
                        self._populate_worker_context(worker_id, self.q.get_nowait(), self.delete_tar_after_extracting)
                    except queue.Empty:
                        # If queue is empty, stop iterator, and thereby this worker.
                        raise StopIteration

                paper_tar_name = self.worker_id_to_context_map[worker_id][PAPER_TAR_NAMES].pop(0)
                paper_tar_processor = PaperTarProcessor(paper_tarname=paper_tar_name, worker_id=worker_id,
                                                        work_dir_prefix=self.work_dir_prefix,
                                                        arxiv_data_output_dir=self.arxiv_data_output_dir,
                                                        augment_typewriter_font=True, augment_line_spacing_1_5=True,
                                                        image_augmentation_transform_sequence=self.image_augmentation_transform_sequence,
                                                        ignore_pages_with_no_figures=self.ignore_pages_with_no_figures)
                try:
                    result_tuple = paper_tar_processor.process_paper_tar()
                    if result_tuple:
                        result_path, figure_boundaries, caption_boundaries = result_tuple
                except Exception:  # Intentional broad catch clause because the try block should ideally throw nothing.
                    logging.warning(
                        'Unhandled exception caught while processing paper tar. Suppressing it and moving forward. Worker ID: {}. paper_tar_name: {}'.format(
                            worker_id, paper_tar_name))
            if self.get_raw_image:
                figure_boundaries = figure_boundaries
            else:
                figure_boundaries = separate_figure_boundaries(figure_boundaries)
            self.worker_id_to_context_map[worker_id][FIGURE_JSONS] = figure_boundaries

        figure_json_retval = self.worker_id_to_context_map[worker_id][FIGURE_JSONS].pop()
        if self.get_raw_image:
            procesed_img, labels = utils.figure_json_to_raw_data(figure_json_retval)
        else:
            procesed_img, labels = utils.figure_json_to_yolo_v3_value(figure_json_retval)
        procesed_img = np.swapaxes(procesed_img, 1, 2)
        procesed_img = np.swapaxes(procesed_img, 0, 1)
        return procesed_img, labels, 0, 0


if __name__ == '__main__':

    if settings.IN_DOCKER:
        logging.basicConfig(filename='/work/host-output/logger_arxiv.log', level=logging.ERROR,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(filename='logger_arxiv.log', level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    input_files = [
        's3://arxiv/src/arXiv_src_1207_004.tar'
    ]

    arxiv_dataset = ArxivDataSet(list_of_files=input_files, image_augmentation_transform_sequence=settings.no_op)
    teacher_train_loader = DataLoader(arxiv_dataset, shuffle=False, num_workers=1)
    for item in teacher_train_loader:
        print("-----------------------------------------------------------")
        print("-----------------------------------------------------------")
        print("-----------------------------------------------------------")
        print(item)
        print("-----------------------------------------------------------")
        print("-----------------------------------------------------------")
        print("-----------------------------------------------------------")
        logging.error(item)

    # figure_boundaries = '[{"image_path": "/work/host-output/arxiv_data_output/diffs_100dpi/0/0003/astro-ph0003102/black.pdf-images/ghostscript/dpi100/black.pdf-dpi100-page0004.png", "rects": [{"x1": 642.9305419921875, "x2": 654.4535522460938, "y1": 370.81756591796875, "y2": 375.594482421875}]}, {"image_path": "/work/host-output/arxiv_data_output/diffs_100dpi/0/0003/astro-ph0003102/black.pdf-images/ghostscript/dpi100/black.pdf-dpi100-page0007.png", "rects": [{"x1": 611.34326171875, "x2": 613.46142578125, "y1": 555.832275390625, "y2": 556.9957885742188}]}, {"image_path": "/work/host-output/arxiv_data_output/diffs_100dpi/0/0003/astro-ph0003102/black.pdf-images/ghostscript/dpi100/black.pdf-dpi100-page0012.png", "rects": [{"x1": 695.3753662109375, "x2": 700.7623901367188, "y1": 424.4481201171875, "y2": 428.1315002441406}]}]'
    # import json
    # figure_boundaries = json.loads(figure_boundaries)
    # separated_figure_boundaries = separate_figure_boundaries(figure_boundaries)
    # assert separated_figure_boundaries == figure_boundaries
    # print(json.dumps(separated_figure_boundaries))
