import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import io


def masktransform(inputimg, maskimg):
    # img = cv2.imread(inputimg, cv2.IMREAD_GRAYSCALE)  # gray image
    img = cv2.imread(inputimg, -1)  # tiff image read original
    maskimg = cv2.imread(maskimg, cv2.IMREAD_GRAYSCALE)
    mask = np.zeros(maskimg.shape)

    rows, cols = img.shape[:2]
    for row in range(rows):
        for col in range(cols):
            if maskimg[row, col] != 0:
                mask[row, col] = 1

    outimg = mask * img
    return outimg


if __name__ == "__main__":

    PATH1 = "./data/deep22/test/masks/"
    PATH2 = "./data/deep0ls/test/masks/"
    PATH3 = "./data/mask5back/test/"

    for i in range(len(os.listdir(PATH1))):

        maskimg = masktransform(PATH1+'%06d'%i+".png", PATH2+'%06d'%i+".png")
        cv2.imwrite(PATH3 + '%06d'%i + ".png", maskimg)
        # cv2.imwrite(PATH3 + '%06d'%i + ".tiff", maskimg, (int(cv2.IMWRITE_TIFF_COMPRESSION), 1))