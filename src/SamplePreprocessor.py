from __future__ import division
from __future__ import print_function

import random
import numpy as np
import cv2
import Config as config


def preprocess(img):
    "scale image into the desired imgSize, transpose it for TF and normalize gray-values"

    # increase dataset size by applying random stretches to the images
    if config.AUGMENT_IMAGE:
        stretch = (random.random() - 0.5)  # -0.5 .. +0.5
        # random width, but at least 1
        wStretched = max(int(img.shape[1] * (1 + stretch)), 1)
        # stretch horizontally by factor 0.5 .. 1.5
        img = cv2.resize(img, (wStretched, img.shape[0]))

    # create target image and copy sample image into it
    (h, w) = img.shape
    fx = w / config.IMAGE_WIDTH
    fy = h / config.IMAGE_HEIGHT
    f = max(fx, fy)
    # scale according to f (result at least 1 and at most wt or ht)
    newSize = (max(min(config.IMAGE_WIDTH, int(w / f)), 1),
               max(min(config.IMAGE_HEIGHT, int(h / f)), 1))
    img = cv2.resize(img, newSize)
    target = np.ones([config.IMAGE_HEIGHT, config.IMAGE_WIDTH]) * 255
    target[0:newSize[1], 0:newSize[0]] = img

    # transpose for TF
    img = cv2.transpose(target)

    # normalize
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s > 0 else img

    return img
