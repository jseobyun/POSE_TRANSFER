import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, downsample=None):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, stride=1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv2 = nn.Conv2d(inplanes, outplanes, stride=stride, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out += residual

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class DiscNet(nn.Module):
    def __init__(self):
        super(DiscNet, self).__init__()
        channels = [64, 128, 256, 256, 64]

        self.inplanes = 256
        self.outplanes = 1
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size= 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size =2 , stride =2, padding = 0)

        self.layer1 = ResBlock(channels[0], channels[1], stride =2)
        self.layer2 = ResBlock(channels[1], channels[2], stride =2)
        self.layer3 = ResBlock(channels[2], channels[3], stride =2)
        self.layer4 = ResBlock(channels[3], channels[4], stride = 2)

        self.final_layer = nn.Linear(64*9*16, self.outplanes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        x = x.view(-1, 64*9*16)
        x = self.final_layer(x)
        x = self.sigmoid(x)
        return x
