import torch
import torch.nn as nn
from torchvision import models
def L1_loss(pred, target):
    loss = torch.abs(pred-target)
    loss = torch.sum(loss, dim=(2, 3))
    loss = torch.mean(loss, dim=(0, 1))

    return loss

def Smooth_L1_loss(pred, target):
    loss = torch.abs(pred-target)
    loss[loss<1] = 0.5*loss[loss<1]**2
    loss[loss>=1] = loss[loss>=1] - 0.5
    loss = torch.sum(loss, dim= (2,3))
    loss = torch.mean(loss, dim=(0,1))
    return loss

def BCE_loss(pred, loss_type):
    B, C = pred.shape
    BCE = torch.nn.BCELoss()
    if loss_type =='fake':
        label = torch.zeros([B,1]).cuda()
    elif loss_type =='real':
        label = torch.ones([B,1]).cuda()
    loss =BCE(pred, label)
    return loss

class VGG16(nn.Module):
    def __init__(self, requires_grad = False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    def forward(self, x):
        h = self.slice1(x)
        h1 = h
        h = self.slice2(h)
        h2 = h
        h = self.slice3(h)
        h3 = h
        h = self.slice4(h)
        h4 = h
        return h1, h2, h3, h4

vgg_model = VGG16().cuda()
def Perceptual_loss(pred, target):
    pred_h1, pred_h2, pred_h3, pred_h4 = vgg_model(pred)
    target_h1, target_h2, target_h3, target_h4 = vgg_model(target)

    loss1 = torch.abs(pred_h1 - target_h1)
    loss2 = torch.abs(pred_h2 - target_h2)
    loss3 = torch.abs(pred_h3 - target_h3)
    loss4 = torch.abs(pred_h4 - target_h4)

    loss1 = torch.mean(loss1, dim=(2, 3))
    loss2 = torch.mean(loss2, dim=(2, 3))
    loss3 = torch.mean(loss3, dim=(2, 3))
    loss4 = torch.mean(loss4, dim=(2, 3))

    loss1 = torch.sum(loss1, dim =1)
    loss2 = torch.sum(loss2, dim=1)
    loss3 = torch.sum(loss3, dim=1)
    loss4 = torch.sum(loss4, dim=1)

    loss = loss1 + loss2 + loss3 + loss4
    loss = torch.mean(loss)
    return loss
