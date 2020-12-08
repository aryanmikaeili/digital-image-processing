import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor




def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
        (rr >= 0) &
        (rr < img.shape[0]) &
        (cc >= 0) &
        (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]


def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img


def find_circle(img):
    # Fill in this function
    return 100, 100, 30


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
        shape0.intersection(shape1).area /
        shape0.union(shape1).area
    )


def main():
    results = []
    for _ in range(1000):
        params, img = noisy_circle(200, 50, 2)
        detected = find_circle(img)
        results.append(iou(params, detected))
    results = np.array(results)
    print((results > 0.7).mean())

train_size = 1000
max_radius = 50
max_noise_level = 2
image_size = 200

Data_X = np.zeros((train_size, image_size, image_size))
Data_y = np.zeros((train_size, 3))

for i in range(train_size):
    noise = np.random.randint(0, max_noise_level)
    image = noisy_circle(image_size, max_radius, noise)
    Data_X[i] = image[1]
    Data_y[i] = np.array(image[0])
    print(i)

split_ratio = 0.2
train_X, train_y, test_X, test_y = train_test_split(Data_X, Data_y, test_size=split_ratio, shuffle=False)

def train_data(Dataset):
    def __init__(self, train_X, train_Y, transform):
        self.train_X = transform(train_X)
        self.train_Y = train_Y
a = 0
a = 0
