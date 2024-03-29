'''
Modify from: https://github.com/usuyama/pytorch-unet
'''

import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class Refine(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_relu = nn.Sequential(
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        self.conv_last = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, raw_alpha, x):
        x = torch.cat([x, raw_alpha], dim=1)
        x = self.conv_relu(x)
        return self.conv_last(x)


class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        pred = self.maxpool(conv1)

        conv2 = self.dconv_down2(pred)
        pred = self.maxpool(conv2)

        conv3 = self.dconv_down3(pred)
        pred = self.maxpool(conv3)

        pred = self.dconv_down4(pred)

        pred = self.upsample(pred)
        pred = torch.cat([pred, conv3], dim=1)

        pred = self.dconv_up3(pred)
        pred = self.upsample(pred)
        pred = torch.cat([pred, conv2], dim=1)

        pred = self.dconv_up2(pred)
        pred = self.upsample(pred)
        pred = torch.cat([pred, conv1], dim=1)

        pred = self.dconv_up1(pred)

        pred_map = self.conv_last(pred)

        return pred_map
