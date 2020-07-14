import os
import glob

data_dir = '/work/cascades/sampanna/yolov5/377269'
test_size = 5000
test_dir = '/work/cascades/sampanna/yolov5/377269/test'
os.makedirs(test_dir)
png_files = glob.glob(os.path.join(data_dir, '*.png'))

for png_path in png_files:
    print('.', end='')
    os.rename(png_path, os.path.join(test_dir, os.path.basename(png_path)))
    anno_path = os.path.join(os.path.dirname(png_path), os.path.basename(png_path).split('.png')[0] + '.txt')
    if os.path.exists(anno_path):
        os.rename(anno_path, os.path.join(test_dir, os.path.basename(anno_path)))
