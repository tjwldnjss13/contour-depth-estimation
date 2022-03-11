import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bn=True, use_activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Sequential()
        self.acti = nn.ELU(inplace=True) if use_activation else nn.Sequential()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.acti(x)

        return x


class Upconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, out_padding=0, use_bn=True, use_activation=True):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, out_padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Sequential()
        self.acti = nn.ELU(inplace=True) if use_activation else nn.Sequential()

    def forward(self, x):
        x = self.upconv(x)
        x = self.bn(x)
        x = self.acti(x)

        return x


class UpconvBilinear(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, scale_factor=2, use_bn=True, use_activation=True):
        super().__init__()
        self.conv = Conv(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Sequential()
        self.acti = nn.ELU(inplace=True) if use_activation else nn.Sequential()
        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        x = self.conv(x)
        x = self.bn(x)
        x = self.acti(x)

        return x