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
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.act = activation() if activation is not None else nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return compose(self.conv, self.bn, self.act)(x)


class Discriminator(nn.Module):
    def __init__(self, image_size, channel_size):
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
        self.fc_in = channel_size * 8 * (image_size // 16) * (image_size // 16)
        self.fc = nn.Linear(self.fc_in, 1)
        # no batchnorm at linear layer
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = compose(self.dn_1, self.dn_2, self.dn_3, self.dn_4)(x)
        x = x.view(-1, self.fc_in)
        return self.act(self.fc(x))
