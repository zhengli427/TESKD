import torch.nn as nn
import math
import torch

__all__ = ['ResNeXt', 'resnext18', 'resnext34', 'resnext50']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def auxiliary_branch(channel_in, channel_out, kernel_size=3):
    layers = []

    layers.append(nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, stride=kernel_size))
    layers.append(nn.BatchNorm2d(channel_out))
    layers.append(nn.ReLU())

    layers.append(nn.Conv2d(channel_out, channel_out, kernel_size=1, stride=1))
    layers.append(nn.BatchNorm2d(channel_out)),
    layers.append(nn.ReLU()),

    return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes * 2, stride)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes * 2, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 2, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes * 2)
        self.conv3 = nn.Conv2d(planes * 2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self, block, layers, num_classes=1000, num_group=32):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], num_group)
        self.layer2 = self._make_layer(block, 128, layers[1], num_group, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], num_group, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], num_group, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

    def _make_layer(self, block, planes, blocks, num_group, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, num_group=num_group))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_group=num_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # x = self.maxpool(x)

        s_out1 = self.layer1(out)
        s_out2 = self.layer2(s_out1)
        s_out3 = self.layer3(s_out2)
        s_out4 = self.layer4(s_out3)

        out = self.avgpool(s_out4)
        final_fea = out
        out = out.view(out.size(0), -1)
        out = self.fc(out)

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


def resnext18(**kwargs):
    """Constructs a ResNeXt-18 model.
    """
    model = ResNeXt(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnext34(**kwargs):
    """Constructs a ResNeXt-34 model.
    """
    model = ResNeXt(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnext50(**kwargs):
    """Constructs a ResNeXt-50 model.
    """
    model = ResNeXt(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


if __name__ == '__main__':
    import torch

    input = torch.ones([8, 3, 32, 32])
    model = resnext18(num_classes=100)
    out = model(input)
    print('test')
