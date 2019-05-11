import torch.nn as nn
from .utils import compose


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
        self.ct = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation() if activation is not None else nn.ReLU(inplace=True)

    def forward(self, x):
        return compose(self.ct, self.bn, self.act)(x)


class Generator(nn.Module):
    def __init__(self, hidden_size, image_size):
        super().__init__()
        self.up_1 = Upsample(hidden_size, image_size * 8, 4, 1, 0)
        self.up_2 = Upsample(image_size * 8, image_size * 4, 4, 2, 1)
        self.up_3 = Upsample(image_size * 4, image_size * 2, 4, 2, 1)
        self.up_4 = Upsample(image_size * 2, image_size, 4, 2, 1)
        self.ct = nn.ConvTranspose2d(image_size, 3, 4, 2, 1, bias=False)
        self.act = nn.Tanh()

    def forward(self, x):
        return compose(self.up_1, self.up_2, self.up_3, self.up_4, self.ct, self.act)(x)
