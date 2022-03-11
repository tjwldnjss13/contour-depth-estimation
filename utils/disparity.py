import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from PIL import Image

from metrics.ssim import SSIM


def predict_left_image_from_right_image_and_left_disparity(right_image, left_disparity):
    device = right_image.device

    batch, _, h, w = right_image.shape
    pixel_ref = torch.linspace(0, w - 1, w).repeat(batch, 3, h, 1).to(device)
    pixels = torch.clamp(pixel_ref - left_disparity, min=0, max=w-1).type(torch.long)

    N = len(right_image.reshape(-1))
    idx1 = torch.linspace(0, batch - 1, batch).repeat(N // batch).reshape(-1, batch).transpose(1, 0).reshape(-1).type(torch.long)
    idx2 = torch.linspace(0, 2, 3).repeat(N // batch // 3).reshape(-1, 3).transpose(1, 0).reshape(-1).repeat(batch).type(torch.long)
    idx3 = torch.linspace(0, h - 1, h).repeat(N // batch // 3 // h).reshape(-1, h).transpose(1, 0).reshape(-1).repeat(batch * 3).type(torch.long)
    idx4 = pixels.reshape(-1).type(torch.long)

    pred_imgl = right_image[idx1, idx2, idx3, idx4].reshape(right_image.shape)

    return pred_imgl


def predict_right_image_from_left_image_and_right_disparity(left_image, right_disparity):
    device = left_image.device

    batch, _, h, w = left_image.shape
    pixel_ref = torch.linspace(0, w - 1, w).repeat(batch, 3, h, 1).to(device)
    pixels = torch.clamp(pixel_ref + right_disparity, min=0, max=w-1).type(torch.long)

    N = len(left_image.reshape(-1))
    idx1 = torch.linspace(0, batch - 1, batch).repeat(N // batch).reshape(-1, batch).transpose(1, 0).reshape(-1).type(torch.long)
    idx2 = torch.linspace(0, 2, 3).repeat(N // batch // 3).reshape(-1, 3).transpose(1, 0).reshape(-1).repeat(batch).type(torch.long)
    idx3 = torch.linspace(0, h - 1, h).repeat(N // batch // 3 // h).reshape(-1, h).transpose(1, 0).reshape(-1).repeat(batch * 3).type(torch.long)
    idx4 = pixels.reshape(-1).type(torch.long)

    pred_imgr = left_image[idx1, idx2, idx3, idx4].reshape(left_image.shape)

    return pred_imgr


def get_left_disparity(left_image, right_image):
    stereo = cv.StereoSGBM_create(minDisparity=64, numDisparities=64, blockSize=5)
    disparity = stereo.compute(left_image, right_image)

    return disparity


def get_right_disparity(left_image, right_image):
    stereo = cv.StereoSGBM_create(minDisparity=64, numDisparities=64, blockSize=5)
    disparity = stereo.compute(right_image, left_image)

    return disparity


def get_image_from_disparity(image, disparity):
    batch_size, _, h, w = image.shape

    x_base = torch.linspace(0, 1, w).repeat(batch_size, h, 1).type_as(image)
    y_base = torch.linspace(0, 1, h).repeat(batch_size, w, 1).transpose(1, 2).type_as(image)

    x_shifts = disparity[:, 0]
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    out = F.grid_sample(image, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros', align_corners=True)

    return out


def get_right_image_from_left_image_right_disparity(image_left, disparity_map_right):
    return get_image_from_disparity(image_left, disparity_map_right)


def get_left_image_from_right_image_left_disparity(image_right, disparity_map_left):
    return get_image_from_disparity(image_right, -disparity_map_left)


def opencv_disparity():
    imgl_pth = '../samples/5_image_left.png'
    imgr_pth = '../samples/5_image_right.png'
    imgl = cv.imread(imgl_pth, 0)
    imgr = cv.imread(imgr_pth, 0)
    dl = get_left_disparity(imgl, imgr)
    dr = get_right_disparity(imgl, imgr)
    dl = torch.Tensor(dl).unsqueeze(0).unsqueeze(0)
    dr = torch.Tensor(dr).unsqueeze(0).unsqueeze(0)
    dl /= 16
    dr /= 16

    print(torch.min(dl), torch.min(dr))
    print(torch.max(dl), torch.max(dr))

    imgl = Image.open(imgl_pth)
    imgr = Image.open(imgr_pth)
    imgl = T.ToTensor()(imgl).unsqueeze(0)
    imgr = T.ToTensor()(imgr).unsqueeze(0)

    pred_imgl = predict_left_image_from_right_image_and_left_disparity(imgr, dl)
    pred_imgr = predict_right_image_from_left_image_and_right_disparity(imgl, dr)

    depthl = .54 * 721 / (1242 * dl)
    depthr = .54 * 721 / (1242 * dr)

    plt.subplot(321)
    plt.imshow(imgl.squeeze().permute(1, 2, 0))
    plt.subplot(322)
    plt.imshow(imgr.squeeze().permute(1, 2, 0))
    plt.subplot(323)
    plt.imshow(pred_imgl.squeeze().permute(1, 2, 0))
    plt.subplot(324)
    plt.imshow(depthl.squeeze().squeeze())
    plt.subplot(325)
    plt.imshow(depthr.squeeze().squeeze())
    plt.subplot(326)
    plt.imshow(pred_imgr.squeeze().permute(1, 2, 0))
    plt.show()


def torch_predict_left_image():
    from PIL import Image

    transform = T.Compose([T.Resize((384, 768)), T.ToTensor()])
    imgl_pth = '../samples/5_image_left.png'
    imgr_pth = '../samples/5_image_right.png'
    dl_pth = '../samples/5_disp_left.png'
    imgl, imgr, dl = Image.open(imgl_pth), Image.open(imgr_pth), Image.open(dl_pth).convert('L')
    dl_og = dl
    imgl, imgr, dl = transform(imgl).unsqueeze(0), transform(imgr).unsqueeze(0), transform(dl).unsqueeze(0)
    pred_imgl = predict_left_image_from_right_image_and_left_disparity_of_left_image(imgl, imgr, dl)
    plt.subplot(321)
    plt.imshow(imgl.squeeze().permute(1, 2, 0))
    plt.subplot(322)
    plt.imshow(imgr.squeeze().permute(1, 2, 0))
    plt.subplot(323)
    plt.imshow(pred_imgl.squeeze().permute(1, 2, 0))
    plt.subplot(324)
    plt.imshow(dl.squeeze())
    plt.subplot(325)
    plt.imshow(dl_og)
    plt.show()


if __name__ == '__main__':
    opencv_disparity()
