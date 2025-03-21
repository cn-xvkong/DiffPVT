import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, in_channel_high, in_channel_low, out_channel):
        super().__init__()
        self.drop_out = 0.1
        in_channel_all = in_channel_low + in_channel_high

        self.conv = nn.Conv2d(in_channel_all, out_channel, kernel_size=3, padding=1)
        self.BN = nn.BatchNorm2d(out_channel)
        self.ReLU = nn.ReLU(True)
        self.dropout = nn.Dropout(self.drop_out)

    def forward(self, low, high):
        while low.size()[2] != high.size()[2]:
            low = F.interpolate(low, scale_factor=2, mode='bilinear')

        fusion = torch.cat([low, high], dim=1)

        # Double_Conv add identity process fusion as a Residual function
        # conv_fusion = self.double_conv1(fusion)
        # SE_fusion = self.SE(conv_fusion)
        # SER_fusion = self.double_conv2(SE_fusion) + self.identity(fusion)
        # output = self.ReLU(SER_fusion)

        # Ablation
        output = self.ReLU(self.BN(self.conv(fusion)))

        output = self.dropout(output)
        return output




