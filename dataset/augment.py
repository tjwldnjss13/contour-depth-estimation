import torch
import cv2 as cv
import numpy as np
import torchvision.transforms as transforms

from PIL import Image

# For test
import matplotlib.pyplot as plt


def stereo_random_hsv(imgl, imgr):
    hsvl = cv.cvtColor(imgl, cv.COLOR_RGB2HSV)
    hsvr = cv.cvtColor(imgr, cv.COLOR_RGB2HSV)

    for i in range(3):
        val = np.random.randint(0, 20)
        prob = np.random.rand()

        if prob < .5:
            hsvl[:, :, i] = cv.add(hsvl[:, :, i], val)
            hsvr[:, :, i] = cv.add(hsvr[:, :, i], val)
        else:
            hsvl[:, :, i] = cv.subtract(hsvl[:, :, i], val)
            hsvr[:, :, i] = cv.subtract(hsvr[:, :, i], val)

    imgl = cv.cvtColor(hsvl, cv.COLOR_HSV2RGB)
    imgr = cv.cvtColor(hsvr, cv.COLOR_HSV2RGB)

    return imgl, imgr


class RandomGaussianNoise(object):
    def __init__(self, mean=0., std=.1, probability=.9):
        self.mean = mean
        self.std = std
        self.prob = probability

    def __call__(self, tensor):
        if np.random.rand() < self.prob:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            return tensor

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std}'


def rotate2d(image, bounding_box, angle):
    """
    :param image: Tensor, [channel, height, width]
    :param bounding_box: Tensor, [num bounding box, (y_min, x_min, y_max, x_max)]
    :param angle: int
    :return: img_rotate, bbox_rotate
    """
    _, h_og, w_og = image.shape
    img = image.permute(1, 2, 0).numpy()
    h, w, _ = img.shape
    x_ctr, y_ctr = int(w / 2), int(h / 2)

    bbox = bounding_box.numpy()

    mat = cv.getRotationMatrix2D((x_ctr, y_ctr), angle, 1)
    abs_cos = abs(mat[0, 0])
    abs_sin = abs(mat[0, 1])

    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)

    mat[0, 2] += bound_w / 2 - x_ctr
    mat[1, 2] += bound_h / 2 - y_ctr

    img_rotate = cv.warpAffine(img, mat, (bound_w, bound_h))

    h_rotate, w_rotate, _ = img_rotate.shape
    x_ctr_rotate, y_ctr_rotate = int(w_rotate / 2), int(h_rotate / 2)

    theta = angle * np.pi / 180
    w_dif, h_dif = int((w_rotate - w) / 2), int((h_rotate - h) / 2)

    bbox_rotate_list = []
    if len(bbox) > 0:
        theta *= -1
        for i in range(len(bbox)):
            # x0, y0, x2, y2 = bbox[i]
            y0, x0, y2, x2 = bbox[i]
            x1, y1, x3, y3 = x2, y0, x0, y2

            # img = cv.circle(img, (x0, y0), 5, (0, 0, 255), thickness=2)
            # img = cv.circle(img, (x1, y1), 5, (0, 0, 255), thickness=2)
            # img = cv.circle(img, (x2, y2), 5, (0, 0, 255), thickness=2)
            # img = cv.circle(img, (x3, y3), 5, (0, 0, 255), thickness=2)
            #
            # img = cv.putText(img, '0', (x0, y0), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
            # img = cv.putText(img, '1', (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
            # img = cv.putText(img, '2', (x2, y2), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
            # img = cv.putText(img, '3', (x3, y3), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
            #
            # plt.imshow(img)
            # plt.show()

            x0, y0, x1, y1 = x0 + w_dif, y0 + h_dif, x1 + w_dif, y1 + h_dif
            x2, y2, x3, y3 = x2 + w_dif, y2 + h_dif, x3 + w_dif, y3 + h_dif

            x0_rot = int((((x0 - x_ctr_rotate) * np.cos(theta)) - ((y0 - y_ctr_rotate) * np.sin(theta)) + x_ctr_rotate))
            y0_rot = int((((x0 - x_ctr_rotate) * np.sin(theta)) + ((y0 - y_ctr_rotate) * np.cos(theta)) + y_ctr_rotate))
            x1_rot = int((((x1 - x_ctr_rotate) * np.cos(theta)) - ((y1 - y_ctr_rotate) * np.sin(theta)) + x_ctr_rotate))
            y1_rot = int((((x1 - x_ctr_rotate) * np.sin(theta)) + ((y1 - y_ctr_rotate) * np.cos(theta)) + y_ctr_rotate))
            x2_rot = int((((x2 - x_ctr_rotate) * np.cos(theta)) - ((y2 - y_ctr_rotate) * np.sin(theta)) + x_ctr_rotate))
            y2_rot = int((((x2 - x_ctr_rotate) * np.sin(theta)) + ((y2 - y_ctr_rotate) * np.cos(theta)) + y_ctr_rotate))
            x3_rot = int((((x3 - x_ctr_rotate) * np.cos(theta)) - ((y3 - y_ctr_rotate) * np.sin(theta)) + x_ctr_rotate))
            y3_rot = int((((x3 - x_ctr_rotate) * np.sin(theta)) + ((y3 - y_ctr_rotate) * np.cos(theta)) + y_ctr_rotate))

            # img_rotate = cv.circle(img_rotate, (x0_rot, y0_rot), 5, (0, 0, 255), thickness=2)
            # img_rotate = cv.circle(img_rotate, (x1_rot, y1_rot), 5, (0, 0, 255), thickness=2)
            # img_rotate = cv.circle(img_rotate, (x2_rot, y2_rot), 5, (0, 0, 255), thickness=2)
            # img_rotate = cv.circle(img_rotate, (x3_rot, y3_rot), 5, (0, 0, 255), thickness=2)
            #
            # img_rotate = cv.putText(img_rotate, '0', (x0_rot, y0_rot), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
            # img_rotate = cv.putText(img_rotate, '1', (x1_rot, y1_rot), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
            # img_rotate = cv.putText(img_rotate, '2', (x2_rot, y2_rot), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
            # img_rotate = cv.putText(img_rotate, '3', (x3_rot, y3_rot), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
            #
            # plt.imshow(img_rotate)
            # plt.show()


            x_min, y_min = int(min(x0_rot, x1_rot, x2_rot, x3_rot)), int(min(y0_rot, y1_rot, y2_rot, y3_rot))
            x_max, y_max = int(max(x0_rot, x1_rot, x2_rot, x3_rot)), int(max(y0_rot, y1_rot, y2_rot, y3_rot))

            bbox_rotate_list.append([y_min, x_min, y_max, x_max])

    h_rot, w_rot, _ = img_rotate.shape
    h_ratio, w_ratio = h_og / h_rot, w_og / w_rot

    img_rotate = cv.resize(img_rotate, (w_og, h_og), interpolation=cv.INTER_CUBIC)

    img_rotate = torch.as_tensor(img_rotate).permute(2, 0, 1)
    bbox_rotate = torch.as_tensor(bbox_rotate_list).type(dtype=torch.float64)

    if len(bbox_rotate) > 0:
        bbox_rotate[:, 0] *= h_ratio
        bbox_rotate[:, 1] *= w_ratio
        bbox_rotate[:, 2] *= h_ratio
        bbox_rotate[:, 3] *= w_ratio

    return img_rotate, bbox_rotate


def rotate2d_with_mask(image, mask):
    _, h_og, w_og = image.shape
    img = image.permute(1, 2, 0).numpy()
    mask = mask.permute(1, 2, 0).numpy()
    h, w, _ = img.shape
    x_ctr, y_ctr = int(w / 2), int(h / 2)

    angle = np.random.randint(-30, 30)

    mat = cv.getRotationMatrix2D((x_ctr, y_ctr), angle, 1)
    abs_cos = abs(mat[0, 0])
    abs_sin = abs(mat[0, 1])

    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)

    mat[0, 2] += bound_w / 2 - x_ctr
    mat[1, 2] += bound_h / 2 - y_ctr

    img_rotate = cv.warpAffine(img, mat, (bound_w, bound_h))
    mask_rotate = cv.warpAffine(mask, mat, (bound_w, bound_h))

    h_rotate, w_rotate, _ = img_rotate.shape

    h_rot, w_rot, _ = img_rotate.shape

    img_rotate = cv.resize(img_rotate, (w_og, h_og), interpolation=cv.INTER_CUBIC)
    mask_rotate = cv.resize(mask_rotate, (w_og, h_og), interpolation=cv.INTER_NEAREST)

    img_rotate = torch.as_tensor(img_rotate).permute(2, 0, 1)
    mask_rotate = torch.as_tensor(mask_rotate).type(dtype=torch.float64)

    return img_rotate, mask_rotate


def rotate2d_augmentation(image, bounding_box=None, mask=None, angle=None):
    additional = []

    _, h_og, w_og = image.shape
    img = image.permute(1, 2, 0).numpy()
    if mask is not None:
        mask = mask.permute(1, 2, 0).numpy()
    h, w, _ = img.shape
    x_ctr, y_ctr = int(w / 2), int(h / 2)

    if angle is None:
        angle = np.random.randint(-10, 10)

    mat = cv.getRotationMatrix2D((x_ctr, y_ctr), angle, 1)
    abs_cos = abs(mat[0, 0])
    abs_sin = abs(mat[0, 1])

    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)

    mat[0, 2] += bound_w / 2 - x_ctr
    mat[1, 2] += bound_h / 2 - y_ctr

    img_rotate = cv.warpAffine(img, mat, (bound_w, bound_h))

    if bounding_box is not None:
        bbox = bounding_box

        h_rotate, w_rotate, _ = img_rotate.shape
        x_ctr_rotate, y_ctr_rotate = int(w_rotate / 2), int(h_rotate / 2)

        theta = angle * np.pi / 180
        w_dif, h_dif = int((w_rotate - w) / 2), int((h_rotate - h) / 2)

        bbox_rotate_list = []
        if len(bbox) > 0:
            theta *= -1
            for i in range(len(bbox)):
                # x0, y0, x2, y2 = bbox[i]
                y0, x0, y2, x2 = bbox[i]
                x1, y1, x3, y3 = x2, y0, x0, y2

                # img = cv.circle(img, (x0, y0), 5, (0, 0, 255), thickness=2)
                # img = cv.circle(img, (x1, y1), 5, (0, 0, 255), thickness=2)
                # img = cv.circle(img, (x2, y2), 5, (0, 0, 255), thickness=2)
                # img = cv.circle(img, (x3, y3), 5, (0, 0, 255), thickness=2)
                #
                # img = cv.putText(img, '0', (x0, y0), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
                # img = cv.putText(img, '1', (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
                # img = cv.putText(img, '2', (x2, y2), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
                # img = cv.putText(img, '3', (x3, y3), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
                #
                # plt.imshow(img)
                # plt.show()

                x0, y0, x1, y1 = x0 + w_dif, y0 + h_dif, x1 + w_dif, y1 + h_dif
                x2, y2, x3, y3 = x2 + w_dif, y2 + h_dif, x3 + w_dif, y3 + h_dif

                x0_rot = int(
                    (((x0 - x_ctr_rotate) * np.cos(theta)) - ((y0 - y_ctr_rotate) * np.sin(theta)) + x_ctr_rotate))
                y0_rot = int(
                    (((x0 - x_ctr_rotate) * np.sin(theta)) + ((y0 - y_ctr_rotate) * np.cos(theta)) + y_ctr_rotate))
                x1_rot = int(
                    (((x1 - x_ctr_rotate) * np.cos(theta)) - ((y1 - y_ctr_rotate) * np.sin(theta)) + x_ctr_rotate))
                y1_rot = int(
                    (((x1 - x_ctr_rotate) * np.sin(theta)) + ((y1 - y_ctr_rotate) * np.cos(theta)) + y_ctr_rotate))
                x2_rot = int(
                    (((x2 - x_ctr_rotate) * np.cos(theta)) - ((y2 - y_ctr_rotate) * np.sin(theta)) + x_ctr_rotate))
                y2_rot = int(
                    (((x2 - x_ctr_rotate) * np.sin(theta)) + ((y2 - y_ctr_rotate) * np.cos(theta)) + y_ctr_rotate))
                x3_rot = int(
                    (((x3 - x_ctr_rotate) * np.cos(theta)) - ((y3 - y_ctr_rotate) * np.sin(theta)) + x_ctr_rotate))
                y3_rot = int(
                    (((x3 - x_ctr_rotate) * np.sin(theta)) + ((y3 - y_ctr_rotate) * np.cos(theta)) + y_ctr_rotate))

                # img_rotate = cv.circle(img_rotate, (x0_rot, y0_rot), 5, (0, 0, 255), thickness=2)
                # img_rotate = cv.circle(img_rotate, (x1_rot, y1_rot), 5, (0, 0, 255), thickness=2)
                # img_rotate = cv.circle(img_rotate, (x2_rot, y2_rot), 5, (0, 0, 255), thickness=2)
                # img_rotate = cv.circle(img_rotate, (x3_rot, y3_rot), 5, (0, 0, 255), thickness=2)
                #
                # img_rotate = cv.putText(img_rotate, '0', (x0_rot, y0_rot), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
                # img_rotate = cv.putText(img_rotate, '1', (x1_rot, y1_rot), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
                # img_rotate = cv.putText(img_rotate, '2', (x2_rot, y2_rot), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
                # img_rotate = cv.putText(img_rotate, '3', (x3_rot, y3_rot), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
                #
                # plt.imshow(img_rotate)
                # plt.show()

                x_min, y_min = int(min(x0_rot, x1_rot, x2_rot, x3_rot)), int(min(y0_rot, y1_rot, y2_rot, y3_rot))
                x_max, y_max = int(max(x0_rot, x1_rot, x2_rot, x3_rot)), int(max(y0_rot, y1_rot, y2_rot, y3_rot))

                bbox_rotate_list.append([y_min, x_min, y_max, x_max])

        h_rot, w_rot, _ = img_rotate.shape
        h_ratio, w_ratio = h_og / h_rot, w_og / w_rot

        bbox_rotate = torch.as_tensor(bbox_rotate_list).type(dtype=torch.float64)

        if len(bbox_rotate) > 0:
            bbox_rotate[:, 0] *= h_ratio
            bbox_rotate[:, 1] *= w_ratio
            bbox_rotate[:, 2] *= h_ratio
            bbox_rotate[:, 3] *= w_ratio

        additional.append(bbox_rotate)

    if mask is not None:
        mask_rotate = cv.warpAffine(mask, mat, (bound_w, bound_h))
        mask_rotate = cv.resize(mask_rotate, (w_og, h_og), interpolation=cv.INTER_NEAREST)
        mask_rotate = torch.as_tensor(mask_rotate).type(dtype=torch.float64)
        additional.append(mask_rotate)

    img_rotate = cv.resize(img_rotate, (w_og, h_og), interpolation=cv.INTER_CUBIC)
    img_rotate = torch.as_tensor(img_rotate).permute(2, 0, 1)

    if len(additional) == 0:
        return img_rotate
    else:
        additional.insert(0, img_rotate)
        return additional


def horizontal_flip(image, bounding_box):
    """
    :param image: Tensor, [channel, height, width]
    :param bounding_box: Tensor, [num bounding box, (y_min, x_min, y_max, x_max)]
    :return:
    """
    img_flip = transforms.RandomHorizontalFlip(1)(image)
    _, h, w = img_flip.shape

    bbox = bounding_box
    for i in range(len(bbox)):
        bbox[i, 1], bbox[i, 3] = w - bbox[i, 3], w - bbox[i, 1]

    return img_flip, bbox


def horizontal_flip_with_mask(image, mask):
    img_flip = transforms.RandomHorizontalFlip(1)(image)
    mask_flip = transforms.RandomHorizontalFlip(1)(mask)

    return img_flip, mask_flip


def horizontal_flip_augmentation(image, bounding_box=None, mask=None):
    additional = []

    prob = np.random.rand()
    if prob > .5:
        img = transforms.RandomHorizontalFlip(1)(image)
        if bounding_box is not None:
            bbox = bounding_box
            if len(bounding_box) > 0:
                bbox_flip = torch.zeros(bbox.shape)
                w = image.shape[-1]
                bbox_flip[..., 0] = bbox[..., 0]
                bbox_flip[..., 1] = w - bbox[..., 3]
                bbox_flip[..., 2] = bbox[..., 2]
                bbox_flip[..., 3] = w - bbox[..., 1]
                additional.append(bbox_flip)
            else:
                additional.append(bbox)
        if mask is not None:
            mask_flip = transforms.RandomHorizontalFlip(1)(mask)
            additional.append(mask_flip)
    else:
        img = image
        if bounding_box is not None:
            additional.append(bounding_box)
        if mask is not None:
            additional.append(mask)

    if len(additional) == 0:
        return img
    else:
        additional.insert(0, img)
        return additional


def shift_with_mask(image, mask):
    dist_y, dist_x = [np.random.randint(-10, 10) for _ in range(2)]

    img_shift = torch.roll(image, shifts=(dist_y, dist_x), dims=(1, 2))
    mask_shift = torch.roll(mask, shifts=(dist_y, dist_x), dims=(1, 2))

    return img_shift, mask_shift


def shift_augmentation(image, bounding_box=None, mask=None):
    additional = []

    dist_y, dist_x = [np.random.randint(-100, 100) for _ in range(2)]
    # if torch.sum(bounding_box[..., 0] + dist_y < 0) + torch.sum(image.shape[-2] < bounding_box[..., 2] + dist_y) > 0:
    #     dist_y = 0
    # if torch.sum(bounding_box[..., 1] + dist_x < 0) + torch.sum(image.shape[-2] < bounding_box[..., 3] + dist_x) > 0:
    #     dist_x = 0
    h_img, w_img = image.shape[1:]

    # print('bbox before : ', bounding_box)
    # print('dist_y before: ', dist_y, ', dist_x before: ', dist_x)

    if len(bounding_box) > 0:
        if torch.sum(bounding_box[..., 0] + dist_y < 0) > 0:
            dist_y = -torch.min(bounding_box[..., 0]).type(torch.int)
            # print('Shift 1')
        elif torch.sum(h_img <= bounding_box[..., 2] + dist_y) > 0:
            dist_y = (h_img - torch.max(bounding_box[..., 2]) - 1).type(torch.int)
            # print('Shift 2')
        if torch.sum(bounding_box[..., 1] + dist_x < 0) > 0:
            dist_x = -torch.min(bounding_box[..., 1]).type(torch.int)
            # print('Shift 3')
        elif torch.sum(w_img <= bounding_box[..., 3] + dist_x) > 0:
            dist_x = (w_img - torch.max(bounding_box[..., 3]) - 1).type(torch.int)
            # print('Shift 4')

    # print('dist_y: ', dist_y, ', dist_x: ', dist_x)

    h_min = max(0, -dist_y)
    h_max = min(image.shape[-2] - dist_y, image.shape[-2])
    w_min = max(0, -dist_x)
    w_max = min(image.shape[-1] - dist_x, image.shape[-1])

    h_shift_min = max(0, dist_y)
    h_shift_max = min(image.shape[-2] + dist_y, image.shape[-2])
    w_shift_min = max(0, dist_x)
    w_shift_max = min(image.shape[-1] + dist_x, image.shape[-1])

    img_shift = torch.zeros(image.shape)
    img_shift[..., h_shift_min:h_shift_max, w_shift_min:w_shift_max] = image[..., h_min:h_max, w_min:w_max]

    # img_shift = torch.roll(image, shifts=(dist_y, dist_x), dims=(1, 2))
    if bounding_box is not None:
        if len(bounding_box) > 0:
            bbox_shift = bounding_box
            bbox_shift[..., 0] += dist_y
            bbox_shift[..., 1] += dist_x
            bbox_shift[..., 2] += dist_y
            bbox_shift[..., 3] += dist_x
            additional.append(bbox_shift)
        else:
            additional.append(bounding_box)
    if mask is not None:
        mask_shift = torch.roll(mask, shifts=(dist_y, dist_x), dims=(1, 2))
        additional.append(mask_shift)

    if len(additional) == 0:
        return image
    else:
        additional.insert(0, img_shift)
        return additional
