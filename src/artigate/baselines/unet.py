"""U-Net (Ronneberger et al., MICCAI 2015) with padded 3x3 convolutions and BatchNorm.

The original architecture: four 2x2 max-pool downsamplings, 64-128-256-512-1024 channels, 2x2
transposed-convolution upsampling, skip connections by concatenation and a 1x1 output convolution.
Two changes, both standard: padding 1 (the output has the input size, no cropping of skips) and
BatchNorm after each convolution. The output is a sigmoid probability, as the losses of the EGE-UNet
recipe (BceDiceLoss) expect.
"""
import torch
from torch import nn


class DoubleConv(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, base_channels=64):
        super().__init__()
        widths = [base_channels * 2 ** i for i in range(5)]  # 64, 128, 256, 512, 1024
        self.encoders = nn.ModuleList([DoubleConv(input_channels, widths[0])])
        self.encoders.extend(DoubleConv(widths[i], widths[i + 1]) for i in range(4))
        self.pool = nn.MaxPool2d(2)
        self.ups = nn.ModuleList(nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2) for i in reversed(range(4)))
        self.decoders = nn.ModuleList(DoubleConv(widths[i] * 2, widths[i]) for i in reversed(range(4)))
        self.head = nn.Conv2d(widths[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for index, encoder in enumerate(self.encoders):
            x = encoder(x if index == 0 else self.pool(x))
            skips.append(x)
        x = skips.pop()
        for up, decoder in zip(self.ups, self.decoders):
            x = decoder(torch.cat([skips.pop(), up(x)], dim=1))
        return torch.sigmoid(self.head(x))
