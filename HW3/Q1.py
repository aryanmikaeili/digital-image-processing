import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def histEqualizer(image):
    dim = image.shape
    image_flat = image.flatten()

    size = dim[0] * dim[1]
    intens, probs = np.unique(image_flat, return_counts=True)
    probs = probs / size
    image_new = np.zeros((size, ))
    for i, pix in enumerate(image_flat):
        intensity_arg = np.where(intens == pix)[0][0] + 1
        new_intensity = np.floor(np.sum(probs[ :intensity_arg]) * 255)
        image_new[i] = new_intensity

    return image_new.reshape((dim[0], dim[1]))


def histMatching(in_image, ref_image):
    dim_image = in_image.shape
    dim_ref = ref_image.shape

    size_image = dim_image[0] * dim_image[1]
    size_ref = dim_ref[0] * dim_ref[1]

    image_flat = in_image.flatten()
    ref_flat = ref_image.flatten()

    image_new = np.zeros((size_image, ))

    intens_image, probs_image = np.unique(image_flat, return_counts=True)
    intens_ref, probs_ref = np.unique(ref_flat, return_counts=True)

    probs_image = probs_image /size_image
    probs_ref = probs_ref / size_ref

    probs_ref = np.cumsum(probs_ref)

    for i, pix in enumerate(image_flat):
        w = np.sum(probs_image[ :np.where(intens_image == pix)[0][0] + 1])
        n = np.where(probs_ref >= w)[0][0]
        image_new[i] = n
        if i % 100 == 0:
            print(i)

    return image_new
image = cv.imread("C:\\Users\\Aryan\\Desktop\\DIP\\HW3\\input_data\\eye.png", cv.IMREAD_GRAYSCALE)
ref =  cv.imread("C:\\Users\\Aryan\\Desktop\\DIP\\HW3\\input_data\\eyeref.png", cv.IMREAD_GRAYSCALE)


histMatching(image, ref)
a = 0