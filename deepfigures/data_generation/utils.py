from __future__ import division
import torch
import numpy as np
import cv2


def preprocess(img, imgsize, jitter, random_placing=False):
    """
    Image preprocess for yolo input
    Pad the shorter side of the image and resize to (imgsize, imgsize)
    Args:
        img (numpy.ndarray): input image whose shape is :math:`(H, W, C)`.
            Values range from 0 to 255.
        imgsize (int): target image size after pre-processing
        jitter (float): amplitude of jitter for resizing
        random_placing (bool): if True, place the image at random position

    Returns:
        img (numpy.ndarray): input image whose shape is :math:`(C, imgsize, imgsize)`.
            Values range from 0 to 1.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
    """
    h, w, _ = img.shape
    img = img[:, :, ::-1]
    assert img is not None

    if jitter > 0:
        # add jitter
        dw = jitter * w
        dh = jitter * h
        new_ar = (w + np.random.uniform(low=-dw, high=dw)) \
                 / (h + np.random.uniform(low=-dh, high=dh))
    else:
        new_ar = w / h

    if new_ar < 1:
        nh = imgsize
        nw = nh * new_ar
    else:
        nw = imgsize
        nh = nw / new_ar
    nw, nh = int(nw), int(nh)

    if random_placing:
        dx = int(np.random.uniform(imgsize - nw))
        dy = int(np.random.uniform(imgsize - nh))
    else:
        dx = (imgsize - nw) // 2
        dy = (imgsize - nh) // 2

    img = cv2.resize(img, (nw, nh))
    sized = np.ones((imgsize, imgsize, 3), dtype=np.uint8) * 127
    sized[dy:dy + nh, dx:dx + nw, :] = img

    info_img = (h, w, nh, nw, dx, dy)
    return sized, info_img


def figure_json_to_yolo_v3_value(figure_json):
    img = cv2.imread(figure_json['image_path'])
    # print("img shape " + str(img.shape))
    # height, width, channels = img.shape
    processed_img, info_img = preprocess(img, 416, jitter=0, random_placing=False)
    rects = figure_json['rects']
    # assert processed_img is not None
    # assert processed_img.shape[0] > 1
    return processed_img, rects_to_labels(rects, info_img[0], info_img[1])


def figure_json_to_raw_data(figure_json):
    img = cv2.imread(figure_json['image_path'])
    rects = figure_json['rects']
    labels = np.zeros((len(rects), 5))
    for idx, rect in enumerate(rects):
        labels[idx, 1] = rect['x1']
        labels[idx, 2] = rect['x2']
        labels[idx, 3] = rect['y1']
        labels[idx, 4] = rect['y2']
    return img, labels


def rects_to_labels(rects, original_height, original_width):
    labels = np.zeros((len(rects), 5))
    # assert len(rects) > 0
    for idx, rect in enumerate(rects):
        x1 = rect['x1']
        x2 = rect['x2']
        y1 = rect['y1']
        y2 = rect['y2']
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        labels[idx, 1] = xc / original_width
        labels[idx, 2] = yc / original_height
        labels[idx, 3] = w / original_width
        labels[idx, 4] = h / original_height
    return labels
