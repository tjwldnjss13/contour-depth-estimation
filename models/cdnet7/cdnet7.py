import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cdnet7.conv import Conv, Upconv
from models.cdnet7.attention_modules import CBAM
from utils.disparity import get_image_from_disparity
from metrics.ssim import ssim


class DisparityPrediction(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = Conv(in_channels, 2, 1, 1, 0, use_bn=False, use_activation=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x) * .3

        return x


class ContourPrediction(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.seq1 = nn.Sequential(
            Conv(in_channels, in_channels//2, 1, 1, 0),
            Conv(in_channels//2, in_channels//4, 3, 1, 1)
        )
        self.seq2 = nn.Sequential(
            Conv(in_channels, in_channels//2, 1, 1, 0),
            Conv(in_channels//2, in_channels//4, 3, 1, 1),
            Conv(in_channels//4, in_channels//4, 3, 1, 1)
        )
        self.conv_last = Conv(in_channels+in_channels//4+in_channels//4, 1, 1, 1, 0, use_bn=False, use_activation=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.seq1(x)
        x2 = self.seq2(x)

        x = torch.cat([x, x1, x2], dim=1)
        x = self.conv_last(x)
        x = self.sigmoid(x)

        return x


class ResidualBlock13(nn.Module):
    def __init__(self, in_channels, activation_last=True):
        super().__init__()
        self.conv = nn.Sequential(
            Conv(in_channels, in_channels//2, 1, 1, 0),
            Conv(in_channels//2, in_channels, 3, 1, 1, use_activation=False)
        )
        self.elu = nn.ELU(inplace=True)
        self.activation_last = activation_last

    def forward(self, x):
        x_conv = self.conv(x)
        x_add = x_conv + x
        if self.activation_last:
            x_add = self.elu(x_add)

        return x_add


def residual_13_layers(in_channels, num_layers, activation_last=True):
    return nn.Sequential(
        *[ResidualBlock13(in_channels) for _ in range(num_layers-1)],
        ResidualBlock13(in_channels, activation_last)
    )


class ResidualBlock131(nn.Module):
    def __init__(self, in_channels, activation_last=True):
        super().__init__()
        self.conv = nn.Sequential(
            Conv(in_channels, in_channels//4, 1, 1, 0),
            Conv(in_channels//4, in_channels//4, 3, 1, 1),
            Conv(in_channels//4, in_channels, 1, 1, 0, use_activation=False)
        )
        self.elu = nn.ELU(inplace=True)
        self.activation_last = activation_last

    def forward(self, x):
        x_conv = self.conv(x)
        x_add = x_conv + x
        if self.activation_last:
            x_add = self.elu(x_add)

        return x_add


def residual_131_layers(in_channels, num_layers, activation_last=True):
    return nn.Sequential(
        *[ResidualBlock131(in_channels) for _ in range(num_layers-1)],
        ResidualBlock131(in_channels, activation_last)
    )


class SRBlock3(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = residual_13_layers(in_channels, 3)
        self.conv2 = nn.Sequential(
            Conv(in_channels, 2*in_channels, 3, 2, 1),
            residual_13_layers(2*in_channels, 3),
            Upconv(2*in_channels, in_channels)
        )
        self.conv3 = nn.Sequential(
            Conv(in_channels, 2*in_channels, 3, 2, 1),
            Conv(2*in_channels, 4*in_channels, 3, 2, 1),
            residual_13_layers(4*in_channels, 3),
            Upconv(4*in_channels, 2*in_channels),
            Upconv(2*in_channels, in_channels)
        )
        self.conv_last = Conv(3*in_channels, in_channels, 1, 1, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x_cat = self.conv_last(torch.cat([x1, x2, x3], dim=1))

        return x_cat


class SRBlock2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = residual_13_layers(in_channels, 3)
        self.conv2 = nn.Sequential(
            Conv(in_channels, 2*in_channels, 3, 2, 1),
            residual_13_layers(2*in_channels, 3),
            Upconv(2*in_channels, in_channels)
        )
        self.conv_last = Conv(2*in_channels, in_channels, 1, 1, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x_cat = self.conv_last(torch.cat([x1, x2], dim=1))

        return x_cat


class CDNet7(nn.Module):
    def __init__(self):
        super().__init__()

        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)

        # ---------------------------- Encoder ---------------------------------
        self.conv1 = nn.Sequential(
            Conv(3, 32, 3, 1, 1),
            CBAM(32)
        )
        self.conv2 = nn.Sequential(
            Conv(32, 64, 3, 2, 1),
            Conv(64, 64, 3, 1, 1),
            CBAM(64)
        )
        self.conv3 = nn.Sequential(
            Conv(64, 128, 3, 2, 1),
            SRBlock3(128),
            CBAM(128)
        )
        self.conv4 = nn.Sequential(
            Conv(128, 256, 3, 2, 1),
            SRBlock3(256),
            CBAM(256)
        )
        self.conv5 = nn.Sequential(
            Conv(256, 512, 3, 2, 1),
            SRBlock2(512),
            CBAM(512)
        )
        self.conv6 = nn.Sequential(
            Conv(512, 1024, 3, 2, 1),
            residual_131_layers(1024, 3)
        )
        self.skip1 = residual_131_layers(32, 1)
        self.skip2 = residual_131_layers(64, 1)
        self.skip3 = residual_131_layers(128, 1)
        self.skip4 = residual_131_layers(256, 1)
        self.skip5 = residual_131_layers(512, 1)
        # ----------------------------------------------------------------------

        # ---------------------------- Decoder ---------------------------------
        self.upconv5 = Upconv(1024, 512)
        self.upconv4 = Upconv(512, 256)
        self.upconv3 = Upconv(256, 128)
        self.upconv2 = Upconv(128, 64)
        self.upconv1 = Upconv(64, 32)

        self.iconv5 = Conv(2 * 512, 512, 3, 1, 1)
        self.iconv4 = Conv(2 * 256, 256, 3, 1, 1)
        self.iconv3 = Conv(2 * 128 + 2, 128, 3, 1, 1)
        self.iconv2 = Conv(2 * 64 + 2, 64, 3, 1, 1)
        self.iconv1 = Conv(2 * 32 + 2 + 1, 32, 3, 1, 1)

        self.iconv_last = nn.Sequential(
            Conv(32 + 1, 16, 1, 1, 0),
            Conv(16, 32, 3, 1, 1),
        )
        # ----------------------------------------------------------------------

        # --------------------------- Disparity --------------------------------
        self.disp4 = DisparityPrediction(256)
        self.disp3 = DisparityPrediction(128)
        self.disp2 = DisparityPrediction(64)
        self.disp1 = DisparityPrediction(32)
        # ----------------------------------------------------------------------

        # ------------------------ Contour prediction --------------------------
        self.cont2 = ContourPrediction(64)
        self.cont1 = ContourPrediction(32)
        # ----------------------------------------------------------------------

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        up5 = self.upconv5(x6)
        skip5 = self.skip5(x5)
        cat5 = torch.cat([up5, skip5], dim=1)
        i5 = self.iconv5(cat5)

        up4 = self.upconv4(i5)
        skip4 = self.skip4(x4)
        cat4 = torch.cat([up4, skip4], dim=1)
        i4 = self.iconv4(cat4)
        disp4 = self.disp4(i4)
        updisp4 = F.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=True)

        up3 = self.upconv3(i4)
        skip3 = self.skip3(x3)
        cat3 = torch.cat([up3, skip3, updisp4], dim=1)
        i3 = self.iconv3(cat3)
        disp3 = self.disp3(i3)
        updisp3 = F.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=True)

        up2 = self.upconv2(i3)
        skip2 = self.skip2(x2)
        cat2 = torch.cat([up2, skip2, updisp3], dim=1)
        i2 = self.iconv2(cat2)
        disp2 = self.disp2(i2)
        updisp2 = F.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=True)
        cont2 = self.cont2(i2)
        upcont2 = F.interpolate(cont2, scale_factor=2, mode='bilinear', align_corners=True)

        up1 = self.upconv1(i2)
        skip1 = self.skip1(x1)
        cat1 = torch.cat([up1, skip1, updisp2, upcont2], dim=1)
        i1 = self.iconv1(cat1)
        cont1 = self.cont1(i1)

        i_last = torch.cat([i1, cont1], dim=1)
        i_last = self.iconv_last(i_last)
        disp1 = self.disp1(i_last)

        return disp1, disp2, disp3, disp4, cont1

    def loss(self, image_left, image_right, disparities, contour_predict, contour_target):
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

            dl_made = get_image_from_disparity(dr, -dl)
            dr_made = get_image_from_disparity(dl, dr)

            loss_l = torch.mean(torch.abs(dl_made - dl))
            loss_r = torch.mean(torch.abs(dr_made - dr))

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

        def get_image_pyramid(image, num_scale):
            images_pyramid = []
            h, w = image.shape[2:]
            for i in range(num_scale):
                h_scale, w_scale = h // (2 ** i), w // (2 ** i)
                images_pyramid.append(F.interpolate(image, size=(h_scale, w_scale), mode='bilinear', align_corners=True))

            return images_pyramid

        alpha_ap = 1
        alpha_ds = 1
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

        loss_cont = F.binary_cross_entropy(contour_predict, contour_target, reduce='mean')

        loss = loss_ap + loss_ds + loss_lr + loss_cont

        return loss, loss_ap.detach().cpu().item(), loss_ds.detach().cpu().item(), loss_lr.detach().cpu().item(), loss_cont.detach().cpu().item()


if __name__ == '__main__':
    import time
    from torchsummary import summary
    cdnet5 = CDNet7().cuda()
    summary(cdnet5, (3, 256, 512))

    Cs = [32, 112, 248, 412, 590]
    Hs = [128, 64, 32, 16, 8]
    Ws = [256, 128, 64, 32, 16]
    Ns = [6, 12, 18, 24, 12]
    growth_rate = 32



    # for i in range(20):
    #     t_start = time.time()
    #     x = torch.ones(1, 3, 256, 512).cuda()
    #     out = model(x)
    #     t_end = time.time()
    #     print(f'{t_end - t_start:.3f}')
















