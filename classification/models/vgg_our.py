'''
VGG16 for CIFAR-10/100 Dataset.

Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['vgg16', 'vgg19']


# cfg = {
#    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }

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


class VGG(nn.Module):
    def __init__(self, num_classes=10, depth=16, dropout=0.0):
        super(VGG, self).__init__()
        self.inplances = 64
        self.conv1 = nn.Conv2d(3, self.inplances, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.inplances)
        self.conv2 = nn.Conv2d(self.inplances, self.inplances, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.inplances)
        self.relu = nn.ReLU(True)
        self.layer1 = self._make_layers(128, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        if depth == 16:
            num_layer = 3
        elif depth == 19:
            num_layer = 4

        self.layer2 = self._make_layers(256, num_layer)
        self.layer3 = self._make_layers(512, num_layer)
        self.layer4 = self._make_layers(512, num_layer)

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            # nn.Dropout(p=dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            # nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

        self.network_channels = [128, 256, 512, 512]
        laterals, upsample = [], []
        for i in range(4):
            laterals.append(self._lateral(self.network_channels[i], 512))
        for i in range(1, 4):
            upsample.append(self._upsample(channels=512))

        self.laterals = nn.ModuleList(laterals)
        self.upsample = nn.ModuleList(upsample)

        self.fuse_1 = nn.Sequential(
            nn.Conv2d(2 * 512, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.fuse_2 = nn.Sequential(
            nn.Conv2d(2 * 512, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.fuse_3 = nn.Sequential(
            nn.Conv2d(2 * 512, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.downsample3 = auxiliary_branch(512, 512, kernel_size=1)
        self.downsample2 = auxiliary_branch(512, 512, kernel_size=2)
        self.downsample1 = auxiliary_branch(512, 512, kernel_size=4)

        self.avg_b1 = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_b2 = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_b3 = nn.AdaptiveAvgPool2d((1, 1))
        # self.avg_b4 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_b1 = nn.Linear(512, num_classes)
        self.fc_b2 = nn.Linear(512, num_classes)
        self.fc_b3 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


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

    def _make_layers(self, input, num_layer):
        layers = []
        for i in range(num_layer):
            conv2d = nn.Conv2d(self.inplances, input, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(input), nn.ReLU(inplace=True)]
            self.inplances = input
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.maxpool(out)

        s_out1 = self.layer1(out)  # 128,8,8
        s_out2 = self.layer2(s_out1)  # 256,4,4
        s_out3 = self.layer3(s_out2)  # 512,2,2
        s_out4 = self.layer4(s_out3)  # 512,1,1

        t_out4 = self.laterals[3](s_out4)  # 128,512,1,1

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

        s_out4 = F.adaptive_avg_pool2d(s_out4, (1, 1))
        final_fea = s_out4
        out = s_out4.view(s_out4.size(0), -1)
        out = self.classifier(out)

        return out, t_out3, t_out2, t_out1, final_fea, b3_fea, b2_fea, b1_fea


def vgg16(pretrained=False, path=None, **kwargs):
    """
    Constructs a VGG16 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = VGG(depth=16, **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model


def vgg19(pretrained=False, path=None, **kwargs):
    """
    Constructs a VGG19 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = VGG(depth=19, **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model


if __name__ == '__main__':
    model = vgg16(num_classes=100)
    input = torch.ones([8, 3, 32, 32])
    output = model(input)
    print('test')
