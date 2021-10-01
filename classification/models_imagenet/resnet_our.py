import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

__all__ = ['resnet18', 'resnet34', 'resnet50']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


# def conv3x3_bn(in_planes, out_planes, stride=1, groups=1):
#     """3x3 convolution with padding"""
#     modules = nn.Sequential(
#         nn.BatchNorm2d(in_planes),
#         nn.ReLU(),
#         nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False),
#     )
#     return modules


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# def conv1x1_bn(in_planes, out_planes, stride=1, groups=1):
#     modules = nn.Sequential(
#         nn.BatchNorm2d(in_planes),
#         nn.ReLU(),
#         nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=False),
#     )
#     return modules


def auxiliary_branch(channel_in, channel_out, kernel_size=3):
    layers = []

    # middle_channel = channel_out // 4

    layers.append(nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, stride=kernel_size))
    layers.append(nn.BatchNorm2d(channel_out))
    layers.append(nn.ReLU())

    layers.append(nn.Conv2d(channel_out, channel_out, kernel_size=1, stride=1))
    layers.append(nn.BatchNorm2d(channel_out)),
    layers.append(nn.ReLU()),

    # layers.append(nn.Conv2d(middle_channel, channel_out, kernel_size=1, stride=1))
    # layers.append(nn.BatchNorm2d(channel_out))
    # layers.append(nn.ReLU())

    return nn.Sequential(*layers)


# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion * planes)
            )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion * planes)
            )

        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.network_channels = [64 * block.expansion, 128 * block.expansion, 256 * block.expansion,
                                 512 * block.expansion]

        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

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

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers = []
        layers.append(block(self.inplanes, planes, stride, self.groups, self.base_width, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _lateral(self, input_size, output_size=512):
        layers = []
        layers.append(nn.Conv2d(input_size, output_size,
                                kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(output_size))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _upsample(self, channels=512):
        layers = []
        layers.append(torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(torch.nn.Conv2d(channels, channels,
                                      kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 64,56,56

        s_out1 = self.layer1(x)  # 64,56,56
        s_out2 = self.layer2(s_out1)  # 128,28,28
        s_out3 = self.layer3(s_out2)  # 256,14,14
        s_out4 = self.layer4(s_out3)  # 512,7,7

        out = self.avgpool(s_out4)
        final_fea = out
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        t_out4 = self.laterals[3](s_out4)  # 512,7,7

        upsample3 = self.upsample[2](t_out4)
        t_out3 = torch.cat([(upsample3 + self.laterals[2](s_out3)), upsample3], dim=1)  # 512 + 512
        t_out3 = self.fuse_3(t_out3)  # 512,14,14

        upsample2 = self.upsample[1](t_out3)
        t_out2 = torch.cat([(upsample2 + self.laterals[1](s_out2)), upsample2], dim=1)  # 512 + 512
        t_out2 = self.fuse_2(t_out2)  # 512,28,28

        upsample1 = self.upsample[0](t_out2)
        t_out1 = torch.cat([(upsample1 + self.laterals[0](s_out1)), upsample1], dim=1)  # 512 + 512
        t_out1 = self.fuse_1(t_out1)  # 512,56,56

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


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if kwargs['num_classes'] != 1000 and pretrained:
        model = ResNet(BasicBlock, [2, 2, 2, 2])
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        model.fc = nn.Linear(model.fc.in_features, kwargs['num_classes'])
        print('done load model')
    else:
        model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if kwargs['num_classes'] != 1000 and pretrained:
        model = ResNet(BasicBlock, [3, 4, 6, 3])
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        model.fc = nn.Linear(model.fc.in_features, kwargs['num_classes'])
        print('done load model')
    else:
        model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if kwargs['num_classes'] != 1000 and pretrained:
        model = ResNet(Bottleneck, [3, 4, 6, 3])
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        model.fc = nn.Linear(model.fc.in_features, kwargs['num_classes'])
        print('done load model')
    else:
        model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


if __name__ == '__main__':
    input = torch.ones([128, 3, 224, 224])
    model = resnet18(num_classes=1000)
    output = model(input)
    print("test")
