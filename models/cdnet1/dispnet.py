import torch
import torch.nn as nn
import torch.nn.functional as F

from models.conv import *
from models.stn import *
from utils.disparity import *


class DisparityPrediction(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = Conv(in_channels, 2, 1, 1, 0, use_activation=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x) * .3

        return x


class DispNet2(nn.Module):
    def __init__(self, img_size=(384, 768)):
        super().__init__()
        # Encoder
        self.conv1 = Conv(3, 64, 7, 2, 3)
        self.conv2 = Conv(64, 128, 3, 2, 1)
        self.conv3 = Conv(128, 256, 3, 2, 1)
        self.conv4 = Conv(256, 512, 3, 2, 1)
        self.conv5 = Conv(512, 512, 3, 2, 1)
        self.conv6 = Conv(512, 1024, 3, 2, 1)

        # Decoder
        self.upconv6 = UpconvBilinear(1024, 512)
        self.upconv5 = UpconvBilinear(512, 512)
        self.upconv4 = UpconvBilinear(512, 256)
        self.upconv3 = UpconvBilinear(256, 128)
        self.upconv2 = UpconvBilinear(128, 64)
        self.upconv1 = UpconvBilinear(64, 32)

        self.iconv6 = Conv(512+512, 512, 3, 1, 1)
        self.iconv5 = Conv(512+512, 512, 3, 1, 1)
        self.iconv4 = Conv(256+256, 256, 3, 1, 1)
        self.iconv3 = Conv(128+128+2, 128, 3, 1, 1)
        self.iconv2 = Conv(64+64+2, 64, 3, 1, 1)
        self.iconv1 = Conv(32+2, 32, 3, 1, 1)

        # Disparity prediction
        self.disp4 = DisparityPrediction(256)
        self.disp3 = DisparityPrediction(128)
        self.disp2 = DisparityPrediction(64)
        self.disp1 = DisparityPrediction(32)

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        skip1 = x1
        skip2 = x2
        skip3 = x3
        skip4 = x4
        skip5 = x5

        # Decoder
        up6 = self.upconv6(x6)
        cat6 = torch.cat([up6, skip5], dim=1)
        i6 = self.iconv6(cat6)

        up5 = self.upconv5(i6)
        cat5 = torch.cat([up5, skip4], dim=1)
        i5 = self.iconv5(cat5)

        up4 = self.upconv4(i5)
        cat4 = torch.cat([up4, skip3], dim=1)
        i4 = self.iconv4(cat4)
        disp4 = self.disp4(i4)
        updisp4 = F.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=True)

        up3 = self.upconv3(i4)
        cat3 = torch.cat([up3, skip2, updisp4], dim=1)
        i3 = self.iconv3(cat3)
        disp3 = self.disp3(i3)
        updisp3 = F.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=True)

        up2 = self.upconv2(i3)
        cat2 = torch.cat([up2, skip1, updisp3], dim=1)
        i2 = self.iconv2(cat2)
        disp2 = self.disp2(i2)
        updisp2 = F.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=True)

        up1 = self.upconv1(i2)
        cat1 = torch.cat([up1, updisp2], dim=1)
        i1 = self.iconv1(cat1)
        disp1 = self.disp1(i1)

        return [disp1, disp2, disp3, disp4]


if __name__ == '__main__':
    from torchsummary import summary
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = DispNet2().to(device)
    x = torch.ones(2, 3, 384, 768).to(device)
    out = model(x)
    for o in out:
        print(o.shape)

















