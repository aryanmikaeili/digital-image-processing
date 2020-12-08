# Q1 script


import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('pic.png', cv2.IMREAD_GRAYSCALE)
plt.imshow(image, cmap='gray')
bit = 7 # Between 1 and 8
requantized_image = (image / (2**(8 - bit))).astype('uint8')
plt.imshow(requantized_image, cmap='gray')


