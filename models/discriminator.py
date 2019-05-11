import torch.nn as nn
from .utils import compose


class Downsample(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        batch_norm=True,
        activation=None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation() if activation is not None else nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return compose(self.conv, self.bn, self.act)(x)


class Discriminator(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.dn_1 = Downsample(3, image_size, 4, 2, 1, batch_norm=False)
        self.dn_2 = Downsample(image_size, image_size * 2, 4, 2, 1)
        self.dn_3 = Downsample(image_size * 2, image_size * 4, 4, 2, 1)
        self.dn_4 = Downsample(image_size * 4, image_size * 8, 4, 2, 1)
        self.conv = nn.Conv2d(image_size * 8, 1, 4, 1, 0, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return compose(self.dn_1, self.dn_2, self.dn_3, self.dn_4, self.conv, self.act)(x)
