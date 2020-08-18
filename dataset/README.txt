Steps to run:
1. Install the dependencies. (requirements.txt or environment.yaml)
2. Install ghostscript on your system. Test if it is installed using 'gs --version'. This code was tested for gs 9.50.
3. Run download_etds.py. The script should take a few minutes to execute depending on the system performance and network download speed. Ignore the GhostScript errors, if any.


List of files in this dataset and their descriptions:

annotations.csv
Contains the raw annotations in CSV format downloaded from the VGG Image Annotator tool, which was used to manually annotate this dataset.

annotations.json
Similar to annotations.csv, this file contains the annotations in JSON format.

etds
This directory contains the PDF and JSON metadata files of all the downloaded ETDs.

figure_boundaries.json
Contains the annotations in COCO object detection dataset format.

figure_boundaries_test.json
A 20% random split of figure_boundaries.json. This file and figure_boundaries_train.json when concatenated will produce figure_boundaries.json.

figure_boundaries_train.json
An 80% random split of figure_boundaries.json. This file and figure_boundaries_test.json when concatenated will produce figure_boundaries.json.

figure_boundaries_testing.json
A 50% random split of figure_boundaries.json. This file and figure_boundaries_validation.json when concatenated will produce figure_boundaries.json.

figure_boundaries_validation.json
A 50% random split of figure_boundaries.json. This file and figure_boundaries_testing.json when concatenated will produce figure_boundaries.json.

images
This is a directory containing all the images referred in the above annotations files. These images have been rendered from their respective PDF files at 100 dpi.

metadata.json
Contains the metadata for all the ETDs used to generate this dataset. The format of this file is pretty self-explanatory.
