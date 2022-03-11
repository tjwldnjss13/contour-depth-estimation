import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.ssim import *
from utils.disparity import *


class LossModule(nn.Module):
    def __init__(self, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        self.device = device
        self.a_ap = 1
        self.a_ds = 1
        self.a_lr = 1

    def appearance_matching_loss(self, image1, image2, alpha=.85):
        """
        :param image1: tensor, [num batches, channels, height, width]
        :param image2: tensor, [num_batches, channels, height, width]
        :param alpha: float, 0~1
        :return:
        """
        assert image1.shape == image2.shape

        N_batch, _, h, w = image1.shape
        N_pixel = h * w

        loss_ssim = alpha * (1 - ssim(image1, image2, 3)) / 2 / N_batch
        loss_l1 = (1 - alpha) * torch.abs(image1 - image2).sum() / N_batch / N_pixel
        loss = loss_ssim + loss_l1

        # print(f' ssim: {ssim(image1, image2, 3).detach().cpu().numpy()} loss_sim: {loss_ssim.detach().cpu().numpy()} \
        #       loss_l1: {loss_l1.detach().cpu().numpy()} loss: {loss.detach().cpu().numpy()}')

        return loss

    def get_image_derivative_x(self, image, filter=None):
        """
        :param image: tensor, [num batches, channels, height, width]
        :param filter: tensor, [num_batches=1, channels, height, width]
        :return:
        """
        if filter is None:
            filter = torch.Tensor([[[[-1, 0, 1],
                                     [-2, 0, 2],
                                     [-1, 0, 1]]]]).to(image.device)

        num_channels = image.shape[1]
        if num_channels > 1:
            filter = torch.cat([filter for _ in range(num_channels)], dim=1)

        derv_x = F.conv2d(image, filter, None, 1, 1)

        return derv_x

    def get_image_derivative_y(self, image, filter=None):
        """
        :param image: tensor, [num batches, channels, height, width]
        :param filter: tensor, [num_batches=1, channels, height, width]
        :return:
        """
        if filter is None:
            filter = torch.Tensor([[[[-1, -2, -1],
                                     [0, 0, 0],
                                     [1, 2, 1]]]]).to(image.device)

        num_channels = image.shape[1]
        if num_channels > 1:
            filter = torch.cat([filter for _ in range(num_channels)], dim=1)

        derv_y = F.conv2d(image, filter, None, 1, 1)

        return derv_y

    def disparity_smoothness_loss(self, image, disparity_map):
        """
        :param image: tensor, [num batches, channels, height, width]
        :param disparity_map: tensor, [num batches, channels, height, width]
        :return:
        """
        img = image
        dmap = disparity_map

        N_batch = image.shape[0]
        N_pixel = image.shape[2] * image.shape[3]

        grad_dmap_x = self.get_image_derivative_x(dmap)
        grad_dmap_y = self.get_image_derivative_y(dmap)

        grad_img_x = self.get_image_derivative_x(img)
        grad_img_y = self.get_image_derivative_y(img)

        grad_img_x = torch.abs(grad_img_x).sum(dim=1).unsqueeze(1)
        grad_img_y = torch.abs(grad_img_y).sum(dim=1).unsqueeze(1)

        loss = (torch.abs(grad_dmap_x) * torch.exp(-torch.abs(grad_img_x)) +
                torch.abs(grad_dmap_y) * torch.exp(-torch.abs(grad_img_y))).sum() / N_pixel / N_batch

        return loss

    def left_right_disparity_consistency_loss(self, disparity_map_left, disparity_map_right):
        assert disparity_map_left.shape == disparity_map_right.shape

        d_l = disparity_map_left
        d_r = disparity_map_right

        d_rl = get_image_from_disparity(d_r, -d_l)
        d_lr = get_image_from_disparity(d_l, d_r)

        loss_l = torch.mean(torch.abs(d_rl - d_l))
        loss_r = torch.mean(torch.abs(d_lr - d_r))

        loss = (loss_l + loss_r).sum()

        # N_batch = dl.shape[0]
        # N_pixel = dl.shape[1] * dl.shape[2]
        #
        # loss_l = torch.zeros(1).to(dl.device)
        # loss_r = torch.zeros(1).to(dl.device)
        #
        # for i in range(dl.shape[1]):
        #     for j in range(dl.shape[2]):
        #         idx_l = j - dl[i, j]
        #         idx_r = j + dl[i, j]
        #
        #         loss_l += torch.abs(dl - dr[:, i, idx_l]).sum()
        #         loss_r += torch.abs(dr - dl[:, i, idx_r]).sum()
        #
        # loss = (loss_l + loss_r) / N_pixel / N_batch

        return loss

    def semantic_segmentation_loss(self, predict, target):
        assert predict.shape == target.shape

        loss = (-target * torch.log2(predict + 1e-15)).mean()

        return loss

    def forward(self, predict_image_left, target_image_left, predict_image_right, target_image_right,
                      disparity_map_left, disparity_map_right, predict_semantic, target_semantic):

        loss_ap = self.a_ap * (self.appearance_matching_loss(predict_image_left, target_image_left) +
                              self.appearance_matching_loss(predict_image_right, target_image_right))
        loss_ds = self.a_ds * (self.disparity_smoothness_loss(target_image_left, disparity_map_left) +
                              self.disparity_smoothness_loss(target_image_right, disparity_map_right))
        loss_lr = self.a_lr * self.left_right_disparity_consistency_loss(disparity_map_left, disparity_map_right)
        loss_sem = self.a_sem * self.semantic_segmentation_loss(predict_semantic, target_semantic)

        loss = loss_ap + loss_ds + loss_lr + loss_sem

        return loss, loss_ap.detach().cpu().item(), loss_ds.detach().cpu().item(), loss_lr.detach().cpu().item(), loss_sem.detach().cpu().item()


def appearance_matching_loss(image1, image2, alpha=.85):
    """
    :param image1: tensor, [num batches, channels, height, width]
    :param image2: tensor, [num_batches, channels, height, width]
    :param alpha: float, 0~1
    :return:
    """
    assert image1.shape == image2.shape

    N_batch, _, h, w = image1.shape
    N_pixel = h * w

    loss_ssim = alpha * ((1 - ssim(image1, image2, 3)) / 2).mean()
    loss_l1 = (1 - alpha) * torch.abs(image1 - image2).mean()
    loss = loss_ssim + loss_l1

    # print(f' ssim: {ssim(image1, image2, 3).detach().cpu().numpy()} loss_sim: {loss_ssim.detach().cpu().numpy()} \
    #       loss_l1: {loss_l1.detach().cpu().numpy()} loss: {loss.detach().cpu().numpy()}')

    return loss


def min_appearance_matching_loss(image1, image2, alpha=.85):
    """
    :param image1: tensor, [num batches, channels, height, width]
    :param image2: tensor, [num_batches, channels, height, width]
    :param alpha: float, 0~1
    :return:
    """
    assert image1.shape == image2.shape

    N_batch, _, h, w = image1.shape
    N_pixel = h * w

    loss_ssim = alpha * ((1 - ssim(image1, image2, 3)) / 2).min()
    loss_l1 = (1 - alpha) * torch.abs(image1 - image2).min()
    loss = loss_ssim + loss_l1

    # print(f' ssim: {ssim(image1, image2, 3).detach().cpu().numpy()} loss_sim: {loss_ssim.detach().cpu().numpy()} \
    #       loss_l1: {loss_l1.detach().cpu().numpy()} loss: {loss.detach().cpu().numpy()}')

    return loss

def get_image_derivative_x(image, filter=None):
    """
    :param image: tensor, [num batches, channels, height, width]
    :param filter: tensor, [num_batches=1, channels, height, width]
    :return:
    """
    if filter is None:
        filter = torch.Tensor([[[[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]]]]).to(image.device)

    num_channels = image.shape[1]
    if num_channels > 1:
        filter = torch.cat([filter for _ in range(num_channels)], dim=1)

    derv_x = F.conv2d(image, filter, None, 1, 1)

    return derv_x

def get_image_derivative_y(image, filter=None):
    """
    :param image: tensor, [num batches, channels, height, width]
    :param filter: tensor, [num_batches=1, channels, height, width]
    :return:
    """
    if filter is None:
        filter = torch.Tensor([[[[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]]]]).to(image.device)

    num_channels = image.shape[1]
    if num_channels > 1:
        filter = torch.cat([filter for _ in range(num_channels)], dim=1)

    derv_y = F.conv2d(image, filter, None, 1, 1)

    return derv_y

def disparity_smoothness_loss(image, disparity_map):
    """
    :param image: tensor, [num batches, channels, height, width]
    :param disparity_map: tensor, [num batches, channels, height, width]
    :return:
    """
    img = image
    dmap = disparity_map

    N_batch = image.shape[0]
    N_pixel = image.shape[2] * image.shape[3]

    grad_dmap_x = get_image_derivative_x(dmap)
    grad_dmap_y = get_image_derivative_y(dmap)

    grad_img_x = get_image_derivative_x(img)
    grad_img_y = get_image_derivative_y(img)

    grad_img_x = torch.abs(grad_img_x).sum(dim=1).unsqueeze(1)
    grad_img_y = torch.abs(grad_img_y).sum(dim=1).unsqueeze(1)

    loss = (torch.abs(grad_dmap_x) * torch.exp(-torch.abs(grad_img_x)) +
            torch.abs(grad_dmap_y) * torch.exp(-torch.abs(grad_img_y))).mean()

    return loss


def left_right_disparity_consistency_loss(disparity_map_left, disparity_map_right):
    assert disparity_map_left.shape == disparity_map_right.shape

    dl = disparity_map_left
    dr = disparity_map_right

    dl_cons = get_image_from_disparity(dr, -dl)
    dr_cons = get_image_from_disparity(dl, dr)

    loss_l = torch.mean(torch.abs(dl_cons - dl))
    loss_r = torch.mean(torch.abs(dr_cons - dr))

    loss = (loss_l + loss_r).sum()

    # N_batch = dl.shape[0]
    # N_pixel = dl.shape[1] * dl.shape[2]
    #
    # loss_l = torch.zeros(1).to(dl.device)
    # loss_r = torch.zeros(1).to(dl.device)
    #
    # for i in range(dl.shape[1]):
    #     for j in range(dl.shape[2]):
    #         idx_l = j - dl[i, j]
    #         idx_r = j + dl[i, j]
    #
    #         loss_l += torch.abs(dl - dr[:, i, idx_l]).sum()
    #         loss_r += torch.abs(dr - dl[:, i, idx_r]).sum()
    #
    # loss = (loss_l + loss_r) / N_pixel / N_batch

    return loss


def semantic_segmentation_loss(predict, target):
    assert predict.shape == target.shape

    loss = (-target * torch.log2(predict + 1e-15)).mean()

    return loss


def dispnet_dl_loss(left_image, predict_left_image, predict_left_disparity):
    device = left_image.device

    alpha_ap = 1
    alpha_ds = 1

    # pred_right_image = predict_right_image_from_left_image_and_right_disparity(left_image, right_image, right_disparity).to(device)
    loss_ap = alpha_ap * appearance_matching_loss(left_image, predict_left_image)
    loss_ds = alpha_ds * disparity_smoothness_loss(left_image, predict_left_disparity)

    loss = loss_ap + loss_ds

    return loss, loss_ap.detach().cpu().item(), loss_ds.detach().cpu().item()


def dispnet_dr_loss(image_left, image_right, disparity_right):
    device = image_left.device

    alpha_ap = 1
    alpha_ds = 1

    pred_imgr = get_image_from_disparity(image_left, disparity_right)
    loss_ap = alpha_ap * appearance_matching_loss(image_right, pred_imgr)
    loss_ds = alpha_ds * disparity_smoothness_loss(image_right, disparity_right)

    loss = loss_ap + loss_ds

    return loss, loss_ap.detach().cpu().item(), loss_ds.detach().cpu().item()


def depthnet_loss(image_left, image_right, disparities):

    def get_image_pyramid(image, num_scale):
        images_pyramid = []
        h, w = image.shape[2:]
        for i in range(num_scale):
            h_scale, w_scale = h // (2 ** i), w // (2 ** i)
            images_pyramid.append(F.interpolate(image, size=(h_scale, w_scale), mode='bilinear', align_corners=True))

        return images_pyramid

    alpha_ap = 1
    alpha_ds = .1
    alpha_lr = 1

    num_scale = 4

    dr_list = [d[:, 0].unsqueeze(1) for d in disparities]
    dl_list = [d[:, 1].unsqueeze(1) for d in disparities]

    imgl_list = get_image_pyramid(image_left, num_scale)
    imgr_list = get_image_pyramid(image_right, num_scale)

    pred_imgr_list = [get_image_from_disparity(imgl_list[i], dr_list[i]) for i in range(num_scale)]
    pred_imgl_list = [get_image_from_disparity(imgr_list[i], -dl_list[i]) for i in range(num_scale)]

    loss_ap = [min_appearance_matching_loss(imgr_list[i], pred_imgr_list[i]) + min_appearance_matching_loss(imgl_list[i], pred_imgl_list[i]) for i in range(num_scale)]
    loss_ds = [disparity_smoothness_loss(imgr_list[i], dr_list[i]) + disparity_smoothness_loss(imgl_list[i], dl_list[i]) for i in range(num_scale)]
    loss_lr = [left_right_disparity_consistency_loss(dr_list[i], dl_list[i]) for i in range(num_scale)]

    loss_ap = alpha_ap * sum(loss_ap)
    loss_ds = alpha_ds * sum(loss_ds)
    loss_lr = alpha_lr * sum(loss_lr)

    loss = loss_ap + loss_ds + loss_lr

    return loss, loss_ap.detach().cpu().item(), loss_ds.detach().cpu().item(), loss_lr.detach().cpu().item()


def total_loss(predict_image_left, target_image_left, predict_image_right, target_image_right,
               disparity_map_left, disparity_map_right, predict_semantic, target_semantic,
               alpha_ap=1, alpha_ds=1, alpha_lr=1, alpha_sem=1):

    loss_ap = alpha_ap * (appearance_matching_loss(predict_image_left, target_image_left) +
                       appearance_matching_loss(predict_image_right, target_image_right))
    loss_ds = alpha_ds * (disparity_smoothness_loss(target_image_left, disparity_map_left) +
                       disparity_smoothness_loss(target_image_right, disparity_map_right))
    loss_lr = alpha_lr * left_right_disparity_consistency_loss(disparity_map_left, disparity_map_right)
    loss_sem = alpha_sem * semantic_segmentation_loss(predict_semantic, target_semantic)

    loss = loss_ap + loss_ds + loss_lr + loss_sem

    # print(loss_ap.detach().cpu().numpy())

    return loss, loss_ap.detach().cpu().item(), loss_ds.detach().cpu().item(), loss_lr.detach().cpu().item(), loss_sem.detach().cpu().item()

# def total_loss_no_jam(predict_image_left, target_image_left, predict_image_right, target_image_right,
#                       disparity_map_left, disparity_map_right,
#                       alpha_ap=1, alpha_ds=1, alpha_lr=1):
#     loss_ap = alpha_ap * (appearance_matching_loss(predict_image_left, target_image_left) +
#                           appearance_matching_loss(predict_image_right, target_image_right))
#     loss_ds = alpha_ds * (disparity_smoothness_loss(target_image_left, disparity_map_left) +
#                           disparity_smoothness_loss(target_image_right, disparity_map_right))
#     loss_lr = alpha_lr * left_right_disparity_consistency_loss(disparity_map_left, disparity_map_right)
#
#     loss = loss_ap + loss_ds + loss_lr
#
#     return loss, loss_ap.detach().cpu().item(), loss_ds.detach().cpu().item(), loss_lr.detach().cpu().item()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image

    img = Image.open('../samples/um_000000_left.png')
    img = np.array(img)
    img = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    print(img.shape)

    loss_module = LossModule()

    grad_x = loss_module.get_image_derivative_x(img)
    grad_y = loss_module.get_image_derivative_y(img)
    print(grad_x.shape)
    print(grad_y.shape)

    plt.figure(0)
    plt.subplot(211)
    plt.imshow(grad_x.squeeze())

    plt.subplot(212)
    plt.imshow(grad_y.squeeze())

    plt.show()

































