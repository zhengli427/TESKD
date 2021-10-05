import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

__all__ = ['cifarresnet18', 'cifarresnet34']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)


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

    layers.append(nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, stride=kernel_size))
    layers.append(nn.BatchNorm2d(channel_out))
    layers.append(nn.ReLU())

    layers.append(nn.Conv2d(channel_out, channel_out, kernel_size=1, stride=1))
    layers.append(nn.BatchNorm2d(channel_out)),
    layers.append(nn.ReLU()),


    return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, groups=1, base_width=64, norm_layer=None):
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
                norm_layer(self.expansion * planes))
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

    def __init__(self, inplanes, planes, stride=1, groups=1, base_width=64, norm_layer=None):
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
                norm_layer(self.expansion * planes))

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


class CIFAR_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, bias=False):
        super(CIFAR_ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.network_channels = [
            64 * block.expansion, 128 * block.expansion, 256 * block.expansion, 512 * block.expansion
        ]

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        laterals, upsample = [], []
        for i in range(4):
            laterals.append(self._lateral(self.network_channels[i], 512 * block.expansion))
        for i in range(1, 4):
            upsample.append(self._upsample(channels=512 * block.expansion))

        self.laterals = nn.ModuleList(laterals)
        self.upsample = nn.ModuleList(upsample)
        self.fuse_1 = nn.Sequential(
            nn.Conv2d(2 * 512 * block.expansion, 512 * block.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512 * block.expansion),
            nn.ReLU(inplace=True),
        )

        self.fuse_2 = nn.Sequential(
            nn.Conv2d(2 * 512 * block.expansion, 512 * block.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512 * block.expansion),
            nn.ReLU(inplace=True),
        )

        self.fuse_3 = nn.Sequential(
            nn.Conv2d(2 * 512 * block.expansion, 512 * block.expansion, kernel_size=1, stride=1, bias=False),
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
        layers.append(torch.nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def _lateral(self, input_size, output_size=512):
        layers = []
        layers.append(nn.Conv2d(input_size, output_size, kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(output_size))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        s_out1 = self.layer1(out)
        s_out2 = self.layer2(s_out1)
        s_out3 = self.layer3(s_out2)
        s_out4 = self.layer4(s_out3)

        out = F.adaptive_avg_pool2d(s_out4, (1, 1))
        final_fea = out
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        t_out4 = self.laterals[3](s_out4)

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


def cifarresnet18(pretrained=False, **kwargs):
    return CIFAR_ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


# def resnet18(pretrained=False, **kwargs):
#
#     if kwargs['num_classes'] != 1000 and pretrained:
#         model = ResNet(BasicBlock, [2, 2, 2, 2])
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
#         model.fc = nn.Linear(model.fc.in_features, kwargs['num_classes'])
#         print('done load model')
#     else:
#         model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
#
#     return model


def cifarresnet34(pretrained=False, **kwargs):
    return CIFAR_ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


# def resnet34(pretrained=False, **kwargs):
#     """Constructs a ResNet-34 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     if kwargs['num_classes'] != 1000 and pretrained:
#         model = ResNet(BasicBlock, [3, 4, 6, 3])
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
#         model.fc = nn.Linear(model.fc.in_features, kwargs['num_classes'])
#         print('done load model')
#     else:
#         model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
#
#     return model


def cifarresnet50(pretrained=False, **kwargs):
    return CIFAR_ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


# def resnet50(pretrained=False, **kwargs):
#     """Constructs a ResNet-50 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     if kwargs['num_classes'] != 1000 and pretrained:
#         model = ResNet(Bottleneck, [3, 4, 6, 3])
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
#         model.fc = nn.Linear(model.fc.in_features, kwargs['num_classes'])
#         print('done load model')
#     else:
#         model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#
#     return model


if __name__ == '__main__':
    input = torch.ones([128, 3, 32, 32])
    model = cifarresnet18(num_classes=1000)
    output = model(input)
    print("test")
