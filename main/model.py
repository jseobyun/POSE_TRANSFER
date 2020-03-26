import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from main.config import cfg

class HeadNet(nn.Module):

    def __init__(self):
        self.inplanes = 256
        self.outplanes = 128

        super(HeadNet, self).__init__()

        self.deconv_layers = self._make_deconv_layer(2)
        self.final_layer = nn.Conv2d(
            in_channels=self.inplanes+3,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def _make_deconv_layer(self, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=self.outplanes,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False))
            layers.append(nn.BatchNorm2d(self.outplanes))
            layers.append(nn.ReLU(inplace=True))

            self.inplanes = self.outplanes

        return nn.Sequential(*layers)

    def forward(self, x, bg_imgs):
        x = self.deconv_layers(x)
        x = torch.cat([x, bg_imgs], dim=1)
        x = self.final_layer(x)
        return x

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes*block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)

class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, num_stacks=cfg.num_stacks, num_blocks=cfg.num_blocks, num_joints=18):
        block = Bottleneck
        super(HourglassNet, self).__init__()

        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.num_joints = num_joints
        self.conv1 = nn.Conv2d(21, self.inplanes, kernel_size=3, stride=2, padding= 1, bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats*block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, 3)) #4
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            if i < num_stacks-1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)

        self.fc_ = nn.ModuleList(fc_)


    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y)
                x = x + fc_

        return x

class TotalFrameWork(nn.Module):
    def __init__(self, main_net, deconv_net):
        super(TotalFrameWork, self).__init__()
        self.main_net = main_net
        self.deconv_net = deconv_net

    def forward(self, input, bg_img):
        out = self.main_net(input)
        out = self.deconv_net(out, bg_img)
        return out

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

def init_weights(m):
    if isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, std=0.001)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0)

from utils.train_utils import load_last_model, load_model
def get_model(mode):
    backbone = HourglassNet()
    head_net = HeadNet()
    model = TotalFrameWork(backbone, head_net)
    if mode =='train':
        if cfg.continue_train:
            model = load_last_model(model, model_type= 'G')
        else:
            backbone.apply(init_weights)
            head_net.apply(init_weights)

        model.train()
    elif mode =='test':
        model = load_model(model, cfg.test_model_epoch, model_type='G')
        model.eval()
    return model.cuda()


def get_discriminator(mode):
    model = DiscNet()
    if mode == 'train':
        if cfg.continue_train:
            model = load_last_model(model, model_type = 'D')
        if not cfg.continue_train:
            model.apply(init_weights)
            model.eval()
    if mode =='test':
        model = load_model(model, cfg.test_model_epoch, model_type= 'D')
        model.eval()
    return model.cuda()