import numpy as np
import cv2
import scipy.ndimage.filters
from scipy.signal import convolve
# from scipy.misc.pilutil import imread
import matplotlib.pyplot as plt
import os

MAX_GRAY_LEVEL_VAL = 255
RGB_REP = 2
GRAYSCALE_REP = 1
RGB_SHAPE = 3


def read_image(fileame, representation):
    """ reads an image file and converts it into a given representation
        representation -  is a code, either 1 or 2 defining whether the
        output should be a grayscale image (1) or an RGB image (2)"""

    if representation == GRAYSCALE_REP:
        return cv2.imread(fileame, 0).astype(np.float64) / MAX_GRAY_LEVEL_VAL
    elif representation == RGB_REP:
        return cv2imread(fileame).astype(np.float64) / MAX_GRAY_LEVEL_VAL


def gaussian_filter(filter_size):
    """ get gaussian filter with filter size"""

    if filter_size == 1:
        return np.array([[1]])

    filter = np.float64(np.array([[1, 1]]))

    for i in range(filter_size - 2):
        filter = scipy.signal.convolve2d(filter, np.array([[1, 1]]))

    return filter / np.sum(filter)


def blur_im(im, filter_vec):
    """ blur an image with filter vector"""
    blur_rows = scipy.ndimage.filters.convolve(im, filter_vec)
    blur_columns = scipy.ndimage.filters.convolve(blur_rows, filter_vec.T)
    return blur_columns


def reduce(im, filter_vec):
    """ reduce an image"""
    blurred_im = blur_im(im, filter_vec)
    return blurred_im[::2, ::2]


def expand(im, filter_vec):
    """ expand an image"""
    # zero padding
    im_shape = im.shape
    expanded_im = np.zeros((2 * im_shape[0], 2 * im_shape[1]))
    expanded_im[::2, ::2] = im
    return blur_im(expanded_im, 2 * filter_vec)


def build_gaussian_pyramid(im, max_levels, filter_size):
    """ functions that construct a Gaussian pyramid of a given image."""

    pyr = []
    filter_vec = gaussian_filter(filter_size)
    next_level_im = im
    max_levels = min(max_levels, int(np.log(im.shape[0] // 16) / np.log(2)) + 1,
                     int(np.log(im.shape[1] // 16) / np.log(2)) + 1)

    for i in range(max_levels):
        pyr.append(next_level_im)
        next_level_im = reduce(np.copy(next_level_im), filter_vec)

    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """ functions that construct a Laplacian pyramid of a given image."""
    pyr = []
    gaus_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)

    for i in range(len(gaus_pyr) - 1):
        Ln = gaus_pyr[i] - expand(np.copy(gaus_pyr[i + 1]), filter_vec)
        pyr.append(Ln)

    pyr.append(gaus_pyr[-1])
    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """ reconstruction of an image from its Laplacian Pyramid."""
    reconstructed_im = lpyr[-1] * coeff[-1]

    for i in range(len(lpyr) - 2, -1, -1):
        reconstructed_im = lpyr[i] * coeff[i] + expand(reconstructed_im, filter_vec)

    return reconstructed_im


def stretch_im(im):
    """ stretching the image to [0,1] values """
    minVal, maxVal = np.min(im), np.max(im)
    stretched_im = np.round(255 * (im - minVal) / (maxVal - minVal))
    return stretched_im


def render_pyramid(pyr, levels):
    """ """

    levels=min(len(pyr),levels)
    new_shape0 = pyr[0].shape[0]
    new_shape1 = int(pyr[0].shape[1] * (1 - np.power(0.5, levels)) / 0.5)
    res = np.zeros((new_shape0, new_shape1))
    start_col = 0
    stretched_im = []

    for level in range(levels):
        stretched_im.append(stretch_im(np.copy(pyr[level])))
        cur_level_shape = stretched_im[level].shape
        res[:cur_level_shape[0], start_col:cur_level_shape[1] + start_col] = stretched_im[level]
        start_col += cur_level_shape[1]
    return res


def display_pyramid(pyr, levels):
    """ render and display the stacked pyramid image"""
    res = render_pyramid(pyr, levels)
    plt.figure()
    plt.imshow(res, cmap="gray")
    plt.show()

def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """ pyramid blending. """
    L1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)[0]
    L2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]

    Gm, filter_vec = build_gaussian_pyramid(np.float64(mask), max_levels, filter_size_mask)

    Lout = np.multiply(Gm, L1) + np.multiply(np.subtract(1, Gm), L2)
    coeff = np.ones(Lout.shape[0])

    return np.clip(laplacian_to_image(Lout, filter_vec, coeff), 0, 1)

