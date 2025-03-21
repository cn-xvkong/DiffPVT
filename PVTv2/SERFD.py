import torch
from torch import nn
import torch.nn.functional as F

# Squeeze and Excitation
class SE(nn.Module):
    def __init__(self, in_channel, out_channel, ratio):
        super(SE, self).__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channel, out_channel // ratio, kernel_size=1)
        self.fc2 = nn.Conv2d(out_channel // ratio, out_channel, kernel_size=1)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x_se = self.global_avg_pool(x)
        x_se = self.relu(self.fc1(x_se))
        x_se = self.sigmoid(self.fc2(x_se))
        x = x_se * x

        return x


# Squeeze-Excitation Residual Fusion Decoder
class SERFD(nn.Module):
    def __init__(self, in_channel_high, in_channel_low, out_channel):
        super().__init__()
        in_channel_all = in_channel_low + in_channel_high
        self.drop_out = 0.1

        self.double_conv1 = nn.Sequential(
            nn.Conv2d(in_channel_all, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel)
        )

        self.double_conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel)
        )

        self.SE = SE(in_channel=out_channel, out_channel=out_channel, ratio=8)

        self.identity = nn.Sequential(
            nn.Conv2d(in_channel_all, out_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channel)
        )

        self.conv = nn.Conv2d(in_channel_all, out_channel, kernel_size=3, padding=1)
        self.BN = nn.BatchNorm2d(out_channel)
        self.ReLU = nn.ReLU(True)
        self.dropout = nn.Dropout(self.drop_out)


    def forward(self, low, high):
        while low.size()[2] != high.size()[2]:
            low = F.interpolate(low, scale_factor=2, mode='bilinear')

        fusion = torch.cat([low, high], dim=1)

        # Double_Conv add identity process fusion as a Residual function
        conv_fusion = self.double_conv1(fusion)
        SE_fusion = self.SE(conv_fusion)
        SER_fusion = self.double_conv2(SE_fusion) + self.identity(fusion)
        output = self.ReLU(SER_fusion)

        # Ablation
        # output = self.ReLU(self.BN(self.conv(fusion)))

        output = self.dropout(output)
        return output
