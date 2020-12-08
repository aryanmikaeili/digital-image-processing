# Q2 script

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


image = cv.imread("mona.png", cv.IMREAD_GRAYSCALE)
image = (image / 2).astype("uint8")
neighboring_image = np.zeros(image.shape, dtype="uint8")
bias = 72
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if j == 0:
            neighboring_image[i][j] = (image[i][j] + bias).astype("uint8")
        else:
            neighboring_image[i][j] = (bias + image[i][j - 1]).astype("uint8") - image[i][j]



current_image = (neighboring_image *1.25).astype("uint8")

a = 0