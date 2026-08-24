import torch
from torch import nn
import torch.nn.functional as F
import math


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        out = F.relu(out)
        return out

class ResNet8_share(nn.Module):
    def __init__(self, in_channels=3):
        super(ResNet8_share, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.layer2 = ResBlk(64, 64, stride=2)
        self.blk2 = ResBlk(64, 128, stride=2)
        self.blk3 = ResBlk(128, 256, stride=2)
        self.blk4 = ResBlk(256, 512, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        return x
class ResNet8_private(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet8_private, self).__init__()
        self.outlayer = nn.Linear(512 * 1 * 1, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        result = {'output': x}
        return result

class ResNet8_entire(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(ResNet8_entire, self).__init__()
        self.type = type
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.layer2 = ResBlk(64, 64, stride=2)
        self.blk2 = ResBlk(64, 128, stride=2)
        self.blk3 = ResBlk(128, 256, stride=2)
        self.blk4 = ResBlk(256, 512, stride=2)
        self.outlayer = nn.Linear(512 * 1 * 1, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        result = {'output': x}
        return result

class VGG16_entire(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(VGG16_entire, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
                                nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
                                nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
                                nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
                                nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
                                nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
                                nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.layer5 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
                               nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
                               nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, 2))
        #self.layer5 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
        #                             nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
        #                             nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Flatten(), nn.Linear(512, 512), nn.ReLU(), nn.Dropout())
        self.fc2 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout())
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        feature = x.view(-1, 512)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        result = {'feature': feature ,'output': x}
        return result


class mobilenet_entire(nn.Module):
    def conv_dw(self, in_channel, out_channel, stride):
        return nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=stride, padding=1, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),

            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def __init__(self, in_channels=3, num_classes=10):
        super(mobilenet_entire, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv_dw2 = self.conv_dw(32, 32, 1)
        self.conv_dw3 = self.conv_dw(32, 64, 2)

        self.conv_dw4 = self.conv_dw(64, 64, 1)
        self.conv_dw5 = self.conv_dw(64, 128, 2)

        self.conv_dw6 = self.conv_dw(128, 128, 1)
        self.conv_dw7 = self.conv_dw(128, 256, 2)

        self.conv_dw8 = self.conv_dw(256, 256, 1)
        self.conv_dw9 = self.conv_dw(256, 512, 2)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv_dw2(out)
        out = self.conv_dw3(out)
        out = self.conv_dw4(out)
        out = self.conv_dw5(out)
        out = self.conv_dw6(out)
        out = self.conv_dw7(out)
        out = self.conv_dw8(out)
        out = self.conv_dw9(out)

        out = F.avg_pool2d(out, 2)
        out = out.view(-1, 512)
        feature = out
        out = self.fc(out)
        result = {'feature': feature, 'output': out}
        return result
