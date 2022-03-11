import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    img_pth = 'samples/house2.jpg'
    img = cv.imread(img_pth, -1)
    h, w = img.shape[:2]

    contour = cv.Canny(img, 255/3, 255) / 255
    contour2 = cv.resize(contour, (w//2, h//2), interpolation=cv.INTER_CUBIC)
    _, contour2 = cv.threshold(contour2, 0, 1, cv.THRESH_BINARY)
    contour3 = cv.resize(contour, (w//4, h//4), interpolation=cv.INTER_CUBIC)
    _, contour3 = cv.threshold(contour3, 0, 1, cv.THRESH_BINARY)
    plt.subplot(221)
    plt.imshow(img)
    plt.subplot(222)
    plt.imshow(contour)
    plt.subplot(223)
    plt.imshow(contour2)
    plt.subplot(224)
    plt.imshow(contour3)
    plt.show()