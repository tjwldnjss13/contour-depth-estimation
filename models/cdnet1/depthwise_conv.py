import torch.nn as nn
import torch.nn.functional as F


class depthwise_conv(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding, use_bn=False, use_activation=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.bn = nn.BatchNorm2d(in_channels) if use_bn else nn.Sequential()
        self.activation = nn.LeakyReLU(inplace=True) if use_activation else nn.Sequential()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.activation(x)

        return x


class depthsize_upconv(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding, output_padding, use_bn=False, use_activation=True):
        super().__init__()
        self.depthwise = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=in_channels)
        self.bn = nn.BatchNorm2d(in_channels) if use_bn else nn.Sequential()
        self.activation = nn.LeakyReLU(inplace=True) if use_activation else nn.Sequential()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.activation(x)

        return x


class pointwise_conv(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=False, use_activation=True):
        super().__init__()
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Sequential()
        self.activation = nn.LeakyReLU(inplace=True) if use_activation else nn.Sequential()

    def forward(self, x):
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.activation(x)

        return x


class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bn=False, use_activation=True):
        super().__init__()
        self.depthwise = depthwise_conv(in_channels, kernel_size, stride, padding, use_bn, use_activation)
        self.pointwise = pointwise_conv(in_channels, out_channels, use_bn, use_activation)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class DSUpconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, use_bn=False, use_activation=True):
        super().__init__()
        self.depthwise = depthsize_upconv(in_channels, in_channels, kernel_size, stride, padding, output_padding)
        # self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, out_padding)
        self.pointwise = pointwise_conv(in_channels, out_channels, use_bn, use_activation)

    def forward(self, x):
        x = self.depthwise(x)
        return x


class DSUpconvBilinear(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=False, use_activation=True):
        super().__init__()
        self.depthwise = depthwise_conv(in_channels, kernel_size, stride, padding, use_bn, use_activation)
        self.pointwise = pointwise_conv(in_channels, out_channels, use_bn, use_activation)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.ConvTranspose2d(3, 16, 3, 1, 1)

    def forward(self, x):
        return self.conv(x)


if __name__ == '__main__':
    from torchsummary import summary
    model1 = Model1().cuda()
    shape = (3, 224, 224)
    model2 = depthwise_separable_conv(3, 16, 3, 1, 1, True, True).cuda()
    summary(model1, shape)
    summary(model2, shape)

