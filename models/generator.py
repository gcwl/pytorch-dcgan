import torch
import torch.nn as nn
from .utils import compose, weights_init


class Upsample(nn.Module):
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
            self.ct = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            )
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.ct = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
            self.bn = None
        self.act = activation() if activation is not None else nn.ReLU(inplace=True)

    def forward(self, x):
        return compose(self.ct, self.bn, self.act)(x)


class Generator(nn.Module):
    def __init__(self, image_size, hidden_size, channel_size, init_weights=None):
        super().__init__()
        self.nc = channel_size * 8
        self.nh = image_size // 16
        self.nw = image_size // 16
        self.fc = nn.Linear(hidden_size, self.nc * self.nh * self.nw)
        # size = (batch_size, channel_size*8, image_size/16, image_size/16)
        self.up_1 = Upsample(channel_size * 8, channel_size * 4, 4, 2, 1)
        # size = (batch_size, channel_size*4, image_size/8, image_size/8)
        self.up_2 = Upsample(channel_size * 4, channel_size * 2, 4, 2, 1)
        # size = (batch_size, channel_size*2, image_size/4, image_size/4)
        self.up_3 = Upsample(channel_size * 2, channel_size, 4, 2, 1)
        # size = (batch_size, channel_size, image_size/2, image_size/2)
        self.up_4 = Upsample(channel_size, 3, 4, 2, 1, batch_norm=False, activation=nn.Tanh)
        # size = (batch_size, 3, image_size, image_size)
        # initialize model weights
        self.apply(init_weights or weights_init)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.nc, self.nh, self.nw)
        return compose(self.up_1, self.up_2, self.up_3, self.up_4)(x)


# Generator(64, 100, 128)
#
# Generator(
#   (fc): Linear(in_features=100, out_features=16384, bias=True)
#   (up_1): Upsample(
#     (ct): ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
#     (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (act): ReLU(inplace)
#   )
#   (up_2): Upsample(
#     (ct): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
#     (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (act): ReLU(inplace)
#   )
#   (up_3): Upsample(
#     (ct): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
#     (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (act): ReLU(inplace)
#   )
#   (up_4): Upsample(
#     (ct): ConvTranspose2d(128, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
#     (act): Tanh()
#   )
# )
