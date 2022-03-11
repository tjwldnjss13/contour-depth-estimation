import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cdnet4.conv import Conv, UpconvBilinear
from models.cdnet4.attention_modules import CBAM
from utils.disparity import get_image_from_disparity
from metrics.ssim import ssim


class BottleneckLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels, 4 * growth_rate, 1, 1, 0),
            nn.BatchNorm2d(4 * growth_rate),
            nn.ELU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, 3, 1, 1),
        )

    def forward(self, x):
        x = torch.cat([x, self.layers(x)], dim=1)

        return x


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layer):
        super().__init__()
        self.num_layer = num_layer
        self.conv_list = nn.Sequential(
            *[BottleneckLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layer)]
        )

    def forward(self, x):
        for n in range(self.num_layer):
            x = self.conv_list[n](x)

        return x


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ELU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        x = self.conv(x)

        return x


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


class CDNet4(nn.Module):
    def __init__(self, growth_rate, reduce_rate):
        super().__init__()
        self.avgpool = nn.AvgPool2d(2, 2)

        # ---------------------------- Encoder ---------------------------------
        in_channels = growth_rate
        self.conv1 = Conv(3, in_channels, 7, 2, 3, True, True)
        self.cbam0 = CBAM(in_channels)
        self.maxpool = nn.MaxPool2d(2, 2)

        num_layer = 6
        self.dense_block1 = DenseBlock(in_channels, growth_rate, num_layer)
        in_channels += num_layer * growth_rate

        out_channels = int(in_channels * reduce_rate)
        self.transition1 = TransitionLayer(in_channels, out_channels)
        in_channels = out_channels

        self.cbam1 = CBAM(in_channels)

        num_layer = 12
        self.dense_block2 = DenseBlock(in_channels, growth_rate, num_layer)
        in_channels += num_layer * growth_rate

        out_channels = int(in_channels * reduce_rate)
        self.transition2 = TransitionLayer(in_channels, out_channels)
        in_channels = out_channels

        self.cbam2 = CBAM(in_channels)

        num_layer = 48
        self.dense_block3 = DenseBlock(in_channels, growth_rate, num_layer)
        in_channels += num_layer * growth_rate

        out_channels = int(in_channels * reduce_rate)
        self.transition3 = TransitionLayer(in_channels, out_channels)
        in_channels = out_channels

        self.cbam3 = CBAM(in_channels)

        num_layer = 32
        self.dense_block4 = DenseBlock(in_channels, growth_rate, num_layer)
        in_channels += num_layer * growth_rate

        out_channels = int(in_channels * reduce_rate)
        self.transition4 = TransitionLayer(in_channels, out_channels)
        in_channels = out_channels

        self.cbam4 = CBAM(in_channels)

        num_layer = 24
        self.dense_block5 = DenseBlock(in_channels, growth_rate, num_layer)
        # ----------------------------------------------------------------------

        # ---------------------------- Decoder ---------------------------------
        self.upconv6 = nn.Sequential(
            UpconvBilinear(1726, 768),
            Conv(768, 768, 3, 1, 1)
        )
        self.upconv5 = nn.Sequential(
            UpconvBilinear(768, 512),
            Conv(512, 512, 3, 1, 1)
        )
        self.upconv4 = nn.Sequential(
            UpconvBilinear(512, 256),
            Conv(256, 256, 3, 1, 1)
        )
        self.upconv3 = nn.Sequential(
            UpconvBilinear(256, 128),
            Conv(128, 128, 3, 1, 1)
        )
        self.upconv2 = nn.Sequential(
            UpconvBilinear(128, 64),
            Conv(64, 64, 3, 1, 1)
        )
        self.upconv1 = nn.Sequential(
            UpconvBilinear(64, 32),
            Conv(32, 32, 3, 1, 1)
        )

        self.iconv6 = Conv(768 + 958, 768, 3, 1, 1)
        self.iconv5 = Conv(512 + 892, 512, 3, 1, 1)
        self.iconv4 = Conv(256 + 248, 256, 3, 1, 1)
        self.iconv3 = Conv(128 + 112 + 2, 128, 3, 1, 1)
        self.iconv2 = Conv(64 + 32 + 2, 64, 3, 1, 1)
        self.iconv1 = Conv(32 + 2 + 1, 32, 3, 1, 1)

        self.iconv_last = nn.Sequential(
            Conv(32 + 1, 8, 1, 1, 0),
            Conv(8, 32, 3, 1, 1),
            *[nn.Sequential(
                Conv(32, 8, 1, 1, 0),
                Conv(8, 32, 3, 1, 1)
            ) for _ in range(4)]
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
        x = self.conv1(x)
        x1 = x = self.cbam0(x)

        x = self.maxpool(x)
        x = self.dense_block1(x)
        x = self.transition1(x)
        x2 = x = self.cbam1(x)

        x = self.avgpool(x)
        x = self.dense_block2(x)
        x = self.transition2(x)
        x3 = x = self.cbam2(x)

        x = self.avgpool(x)
        x = self.dense_block3(x)
        x = self.transition3(x)
        x4 = x = self.cbam3(x)

        x = self.avgpool(x)
        x = self.dense_block4(x)
        x = self.transition4(x)
        x5 = x = self.cbam4(x)

        x = self.avgpool(x)
        x6 = x = self.dense_block5(x)

        up6 = self.upconv6(x6)
        cat6 = torch.cat([up6, x5], dim=1)
        i6 = self.iconv6(cat6)

        up5 = self.upconv5(i6)
        cat5 = torch.cat([up5, x4], dim=1)
        i5 = self.iconv5(cat5)

        up4 = self.upconv4(i5)
        cat4 = torch.cat([up4, x3], dim=1)
        i4 = self.iconv4(cat4)
        disp4 = self.disp4(i4)
        updisp4 = F.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=True)

        up3 = self.upconv3(i4)
        cat3 = torch.cat([up3, x2, updisp4], dim=1)
        i3 = self.iconv3(cat3)
        disp3 = self.disp3(i3)
        updisp3 = F.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=True)

        up2 = self.upconv2(i3)
        cat2 = torch.cat([up2, x1, updisp3], dim=1)
        i2 = self.iconv2(cat2)
        disp2 = self.disp2(i2)
        updisp2 = F.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=True)
        cont2 = self.cont2(i2)
        upcont2 = F.interpolate(cont2, scale_factor=2, mode='bilinear', align_corners=True)

        up1 = self.upconv1(i2)
        cat1 = torch.cat([up1, updisp2, upcont2], dim=1)
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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CDNet4(32, .5).to(device)
    summary(model, (3, 256, 512))

    # for i in range(10):
    #     t_start = time.time()
    #     x = torch.ones(1, 3, 256, 512).cuda()
    #     out = model(x)
    #     t_end = time.time()
    #     print(f'{t_end - t_start:.3f}')

















