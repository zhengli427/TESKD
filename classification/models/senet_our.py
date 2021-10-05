"""senet in pytorch
[1] Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu
    Squeeze-and-Excitation Networks
    https://arxiv.org/abs/1709.01507
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def auxiliary_branch(channel_in, channel_out, kernel_size=3):
    layers = []

    layers.append(nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, stride=kernel_size))
    layers.append(nn.BatchNorm2d(channel_out))
    layers.append(nn.ReLU())

    layers.append(nn.Conv2d(channel_out, channel_out, kernel_size=1, stride=1))
    layers.append(nn.BatchNorm2d(channel_out)),
    layers.append(nn.ReLU()),

    return nn.Sequential(*layers)


class BasicResidualSEBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, r=16):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * self.expansion, 3, padding=1),
            nn.BatchNorm2d(out_channels * self.expansion),
            nn.ReLU(inplace=True)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(out_channels * self.expansion, out_channels * self.expansion // r),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion),
            nn.Sigmoid()
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)

        squeeze = self.squeeze(residual)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)

        x = residual * excitation.expand_as(residual) + shortcut

        return F.relu(x)


class BottleneckResidualSEBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, r=16):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * self.expansion, 1),
            nn.BatchNorm2d(out_channels * self.expansion),
            nn.ReLU(inplace=True)
        )

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(out_channels * self.expansion, out_channels * self.expansion // r),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion),
            nn.Sigmoid()
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)

        residual = self.residual(x)
        squeeze = self.squeeze(residual)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)

        x = residual * excitation.expand_as(residual) + shortcut

        return F.relu(x)


class SEResNet(nn.Module):

    def __init__(self, block, block_num, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage1 = self._make_stage(block, block_num[0], 64, 1)
        self.stage2 = self._make_stage(block, block_num[1], 128, 2)
        self.stage3 = self._make_stage(block, block_num[2], 256, 2)
        self.stage4 = self._make_stage(block, block_num[3], 512, 2)

        self.linear = nn.Linear(self.in_channels, num_classes)
        self.network_channels = [64 * block.expansion, 128 * block.expansion, 256 * block.expansion,
                                 512 * block.expansion]

        laterals, upsample = [], []
        for i in range(4):
            laterals.append(self._lateral(self.network_channels[i], 512 * block.expansion))
        for i in range(1, 4):
            upsample.append(self._upsample(channels=512 * block.expansion))
        self.laterals = nn.ModuleList(laterals)
        self.upsample = nn.ModuleList(upsample)

        self.fuse_1 = nn.Sequential(
            nn.Conv2d(2 * 512 * block.expansion, 512 * block.expansion, kernel_size=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(512 * block.expansion),
            nn.ReLU(inplace=True),
        )

        self.fuse_2 = nn.Sequential(
            nn.Conv2d(2 * 512 * block.expansion, 512 * block.expansion, kernel_size=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(512 * block.expansion),
            nn.ReLU(inplace=True),
        )

        self.fuse_3 = nn.Sequential(
            nn.Conv2d(2 * 512 * block.expansion, 512 * block.expansion, kernel_size=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(512 * block.expansion),
            nn.ReLU(inplace=True),
        )

        self.downsample3 = auxiliary_branch(512 * block.expansion, 512 * block.expansion, kernel_size=2)
        self.downsample2 = auxiliary_branch(512 * block.expansion, 512 * block.expansion, kernel_size=4)
        self.downsample1 = auxiliary_branch(512 * block.expansion, 512 * block.expansion, kernel_size=8)

        self.avg_b1 = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_b2 = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_b3 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_b1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc_b2 = nn.Linear(512 * block.expansion, num_classes)
        self.fc_b3 = nn.Linear(512 * block.expansion, num_classes)

    def _upsample(self, channels=512):
        layers = []
        layers.append(torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(torch.nn.Conv2d(channels, channels,
                                      kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def _lateral(self, input_size, output_size=512):
        layers = []
        layers.append(nn.Conv2d(input_size, output_size,
                                kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(output_size))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.pre(x)

        s_out1 = self.stage1(out)
        s_out2 = self.stage2(s_out1)
        s_out3 = self.stage3(s_out2)
        s_out4 = self.stage4(s_out3)

        out = F.adaptive_avg_pool2d(s_out4, 1)
        final_fea = out
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        t_out4 = self.laterals[3](s_out4)  # 128,512,4,4

        upsample3 = self.upsample[2](t_out4)
        t_out3 = torch.cat([(upsample3 + self.laterals[2](s_out3)), upsample3], dim=1)  # 512 + 512
        t_out3 = self.fuse_3(t_out3)  # 512

        upsample2 = self.upsample[1](t_out3)
        t_out2 = torch.cat([(upsample2 + self.laterals[1](s_out2)), upsample2], dim=1)  # 512 + 512
        t_out2 = self.fuse_2(t_out2)  # 512

        upsample1 = self.upsample[0](t_out2)
        t_out1 = torch.cat([(upsample1 + self.laterals[0](s_out1)), upsample1], dim=1)  # 512 + 512
        t_out1 = self.fuse_1(t_out1)

        t_out3 = self.downsample3(t_out3)
        t_out3 = self.avg_b3(t_out3)
        b3_fea = t_out3
        t_out3 = torch.flatten(t_out3, 1)
        t_out3 = self.fc_b3(t_out3)

        t_out2 = self.downsample2(t_out2)
        t_out2 = self.avg_b2(t_out2)
        b2_fea = t_out2
        t_out2 = torch.flatten(t_out2, 1)
        t_out2 = self.fc_b2(t_out2)

        t_out1 = self.downsample1(t_out1)
        t_out1 = self.avg_b1(t_out1)
        b1_fea = t_out1
        t_out1 = torch.flatten(t_out1, 1)
        t_out1 = self.fc_b1(t_out1)

        return out, t_out3, t_out2, t_out1, final_fea, b3_fea, b2_fea, b1_fea

    def _make_stage(self, block, num, out_channels, stride):

        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion

        while num - 1:
            layers.append(block(self.in_channels, out_channels, 1))
            num -= 1

        return nn.Sequential(*layers)


def seresnet18(**kwargs):
    return SEResNet(BasicResidualSEBlock, [2, 2, 2, 2], **kwargs)


def seresnet34(**kwargs):
    return SEResNet(BasicResidualSEBlock, [3, 4, 6, 3], **kwargs)


def seresnet50(**kwargs):
    return SEResNet(BottleneckResidualSEBlock, [3, 4, 6, 3], **kwargs)
