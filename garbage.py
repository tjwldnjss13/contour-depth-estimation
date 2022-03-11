import cv2 as cv
import matplotlib.pyplot as plt

from metrics.ssim import *


class model1(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


if __name__ == '__main__':
    sample_list = [
        f'./samples/contour/{i}_left.png' for i in range(1, 7)
    ]
    for img_pth in sample_list:
        img = cv.imread(img_pth)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        contour = cv.Canny(img, 255 / 3, 255)
        contour = contour > 0
        contour = 1 - contour
        plt.imshow(contour, cmap='gray')
        plt.show()