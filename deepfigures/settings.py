"""Constants and settings for deepfigures."""

import logging
import os
import socket

from imgaug import augmenters as iaa

logger = logging.getLogger(__name__)

HOSTNAME = socket.gethostname()

IN_DOCKER = os.environ.get('IN_DOCKER', False)
IN_IR = os.environ.get('HOSTNAME', 'local') is 'ir.cs.vt.edu' or HOSTNAME == 'ir.cs.vt.edu'
ARC_CLUSTERS = ['cascades', 'newriver', 'dragonstooth', 'huckleberry']
IN_ARC = os.environ.get('SYSNAME', 'local') in ARC_CLUSTERS
ECE_HOSTNAMES = ['big.lan.ece', 'cluster01', 'cluster02', 'cluster03', 'cluster04', 'cluster05', 'cluster06',
                 'cluster07', 'cluster08', 'cluster09', 'cluster10', 'cluster11', 'cluster12', 'cluster13', 'cluster14',
                 'cluster15']
IN_ECE = os.environ.get('HOSTNAME', 'local') in ECE_HOSTNAMES

# path to the deepfigures project root
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__)))

# version number for the current release
VERSION = '0.0.6'

# descriptions of the docker images deepfigures builds
DEEPFIGURES_IMAGES = {
    'cpu': {
        'tag': 'sampyash/vt_cs_6604_digital_libraries',
        'dockerfile_path': os.path.join(BASE_DIR, 'dockerfiles/cpu/Dockerfile'),
        'version_prefix': 'deepfigures_cpu_'
    },
    'gpu': {
        'tag': 'sampyash/vt_cs_6604_digital_libraries',
        'dockerfile_path': os.path.join(BASE_DIR, 'dockerfiles/gpu/Dockerfile'),
        'version_prefix': 'deepfigures_gpu_'
    }
}

# path to the directory containing all the project-level test data.
TEST_DATA_DIR = os.path.join(BASE_DIR, 'tests/data')

# settings for PDFRenderers
DEFAULT_INFERENCE_DPI = 100
DEFAULT_CROPPED_IMG_DPI = 200
BACKGROUND_COLOR = 255

# weights for the model
TENSORBOX_MODEL = {
    'save_dir': os.path.join(BASE_DIR, 'weights/'),
    'iteration': 500000
}

# paths to binary dependencies
PDFFIGURES_JAR_NAME = 'pdffigures2-assembly-0.1.0.jar'
PDFFIGURES_JAR_PATH = os.path.join(
    BASE_DIR,
    'bin/',
    PDFFIGURES_JAR_NAME)

# PDF Rendering backend settings
DEEPFIGURES_PDF_RENDERER = 'deepfigures.extraction.renderers.GhostScriptRenderer'

# settings for data generation

if IN_DOCKER:
    # The location to temporarily store arxiv source data
    ARXIV_DATA_TMP_DIR = '/work/host-output/arxiv_data_temp'
    # The location to store the final output labels
    ARXIV_DATA_OUTPUT_DIR = '/work/host-output/arxiv_data_output'
    ARXIV_DATA_CACHE_DIR = '/work/host-output/download_cache'
    # Higher parallelism on docker.
    PROCESS_PAPER_TAR_THREAD_COUNT = 2 * os.cpu_count()
    # List of tar file to process
    FILE_LIST = '/work/host-input/files.json'
    PDFLATEX_EXECUTABLE_PATH = 'pdflatex'
elif IN_ARC:
    # The location to temporarily store arxiv source data
    ARXIV_DATA_TMP_DIR = '/work/' + os.environ.get('SYSNAME',
                                                   'cascades') + '/sampanna/deepfigures-results/arxiv_data_temp'
    # The location to store the final output labels
    ARXIV_DATA_OUTPUT_DIR = '/work/' + os.environ.get('SYSNAME',
                                                      'cascades') + '/sampanna/deepfigures-results/arxiv_data_output'
    ARXIV_DATA_CACHE_DIR = '/work/' + os.environ.get('SYSNAME',
                                                     'cascades') + '/sampanna/deepfigures-results/download_cache'
    # Lower parallelism on local for simpler debugging.
    PROCESS_PAPER_TAR_THREAD_COUNT = 2 * os.cpu_count()
    # List of tar file to process
    FILE_LIST = '/work/' + os.environ.get('SYSNAME',
                                          'cascades') + '/sampanna/deepfigures-results/files.json'
    PDFLATEX_EXECUTABLE_PATH = '/home/sampanna/texlive/install-tl-20200425/texlive/installation/2020/bin/x86_64-linux/pdflatex'
elif IN_ECE:
    # The location to temporarily store arxiv source data
    ARXIV_DATA_TMP_DIR = '/home/sampanna/deepfigures-results/arxiv_data_temp'
    # The location to store the final output labels
    ARXIV_DATA_OUTPUT_DIR = '/home/sampanna/deepfigures-results/arxiv_data_output'
    ARXIV_DATA_CACHE_DIR = '/home/sampanna/deepfigures-results/download_cache'
    # Lower parallelism on local for simpler debugging.
    PROCESS_PAPER_TAR_THREAD_COUNT = 2 * os.cpu_count()
    # List of tar file to process
    FILE_LIST = '/home/sampanna/deepfigures-results/files.json'
    PDFLATEX_EXECUTABLE_PATH = '/home/sampanna/texlive/2020/bin/x86_64-linux/pdflatex'
elif IN_IR:
    # The location to temporarily store arxiv source data
    ARXIV_DATA_TMP_DIR = '/home/sampanna/deepfigures-results/arxiv_data_temp'
    # The location to store the final output labels
    ARXIV_DATA_OUTPUT_DIR = '/home/sampanna/deepfigures-results/arxiv_data_output'
    ARXIV_DATA_CACHE_DIR = '/home/sampanna/deepfigures-results/download_cache'
    # Lower parallelism on local for simpler debugging.
    PROCESS_PAPER_TAR_THREAD_COUNT = 2 * os.cpu_count()
    # List of tar file to process
    FILE_LIST = '/home/sampanna/deepfigures-results/files.json'
    PDFLATEX_EXECUTABLE_PATH = '/home/sampanna/texlive/2020/bin/x86_64-linux/pdflatex'
else:
    # The location to temporarily store arxiv source data
    ARXIV_DATA_TMP_DIR = '/home/sampanna/workspace/bdts2/deepfigures-results/arxiv_data_temp'
    # The location to store the final output labels
    ARXIV_DATA_OUTPUT_DIR = '/home/sampanna/workspace/bdts2/deepfigures-results/arxiv_data_output'
    ARXIV_DATA_CACHE_DIR = '/home/sampanna/workspace/bdts2/deepfigures-results/download_cache'
    # Lower parallelism on local for simpler debugging.
    PROCESS_PAPER_TAR_THREAD_COUNT = 1
    # List of tar file to process
    FILE_LIST = '/home/sampanna/workspace/bdts2/deepfigures-results/files.json'
    PDFLATEX_EXECUTABLE_PATH = 'pdflatex'

# The location of the PMC open access data
PUBMED_INPUT_DIR = ''
# A directory for storing intermediate results
PUBMED_INTERMEDIATE_DIR = ''
# A directory for storing the output pubmed data
PUBMED_DISTANT_DATA_DIR = ''

# a local directory for storing the output data
LOCAL_PUBMED_DISTANT_DATA_DIR = ''

seq = iaa.Sequential([
    iaa.Affine(rotate=(-5, 5)),
    iaa.AdditiveGaussianNoise(scale=(10, 60)),
    iaa.SaltAndPepper(p=0.1),
    iaa.GaussianBlur(sigma=0.5),
    iaa.LinearContrast(alpha=1),
    iaa.PerspectiveTransform(scale=0.025, keep_size=True)
])

no_op = iaa.Sequential([
    iaa.Fliplr(p=1),
    iaa.Fliplr(p=1)
])
