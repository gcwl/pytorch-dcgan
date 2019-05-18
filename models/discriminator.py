import torch
import torch.nn as nn
from .utils import compose, weights_init


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
        if batch_norm:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            )
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.bn = None
        self.act = activation() if activation is not None else nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return compose(self.conv, self.bn, self.act)(x)


class Discriminator(nn.Module):
    def __init__(self, image_size, channel_size, init_weights=None):
        super().__init__()
        # no batchnorm at first layer
        self.dn_1 = Downsample(3, channel_size, 4, 2, 1, batch_norm=False)
        # size = (batch_size, channel_size, image_size/2, image_size/2)
        self.dn_2 = Downsample(channel_size, channel_size * 2, 4, 2, 1)
        # size = (batch_size, channel_size*2, image_size/4, image_size/4)
        self.dn_3 = Downsample(channel_size * 2, channel_size * 4, 4, 2, 1)
        # size = (batch_size, channel_size*4, image_size/8, image_size/8)
        self.dn_4 = Downsample(channel_size * 4, channel_size * 8, 4, 2, 1)
        # size = (batch_size, channel_size*8, image_size/16, image_size/16)
        self.affine_in = channel_size * 8 * (image_size // 16) * (image_size // 16)
        self.affine = nn.Linear(self.affine_in, 1)
        # no batchnorm at linear layer
        self.act = nn.Sigmoid()
        # initialize model weights
        self.apply(init_weights or weights_init)

    def forward(self, x):
        x = compose(self.dn_1, self.dn_2, self.dn_3, self.dn_4)(x)
        x = x.view(-1, self.affine_in)
        return self.act(self.affine(x))


# Discriminator(64, 128)
#
# Discriminator(
#   (dn_1): Downsample(
#     (conv): Conv2d(3, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
#     (act): LeakyReLU(negative_slope=0.2, inplace)
#   )
#   (dn_2): Downsample(
#     (conv): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
#     (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (act): LeakyReLU(negative_slope=0.2, inplace)
#   )
#   (dn_3): Downsample(
#     (conv): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
#     (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (act): LeakyReLU(negative_slope=0.2, inplace)
#   )
#   (dn_4): Downsample(
#     (conv): Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
#     (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (act): LeakyReLU(negative_slope=0.2, inplace)
#   )
#   (fc): Linear(in_features=16384, out_features=1, bias=True)
#   (act): Sigmoid()
# )
