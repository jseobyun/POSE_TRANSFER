import torch
import torch.nn as nn
import torch.nn.functional as F
from main.config import cfg
from utils.loss_utils import GAN_loss_calculator, vgg_model, mssim_calculator

def L1_loss(pred, target):
    criterion = torch.nn.L1Loss()
    loss =criterion(pred,target)
    return loss

def Smooth_L1_loss(pred, target):
    criterion = torch.nn.SmoothL1Loss()
    loss =criterion(pred,target)
    return loss

def Intermediate_loss(intermediates, target):
    criterion = torch.nn.L1Loss()
    target_resized = F.interpolate(target.clone(), (72,128))
    loss = sum(criterion(intermediate, target_resized) for intermediate in intermediates)
    return loss

def GAN_loss(pred, loss_type):
    if loss_type == 'real':
        label = True
    elif loss_type == 'fake':
        label = False

    loss = GAN_loss_calculator(pred, label)

    return loss

def FM_loss(fake, real):
    criterion = torch.nn.L1Loss()
    feat_weights = 4.0 / (cfg.n_layers_D +1)
    D_weights = 1.0 / cfg.num_D
    loss = 0
    for i in range(cfg.num_D):
        for j in range(len(fake[i])): # -1 deleted
            loss += D_weights * feat_weights * criterion(fake[i][j], real[i][j].detach())
    return loss


def Perceptual_loss(pred, target):
    criterion = torch.nn.L1Loss()

    weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
    pred_vgg = vgg_model(pred)
    target_vgg = vgg_model(target)

    loss = 0
    for i in range(len(pred_vgg)):
        loss += weights[i] *criterion(pred_vgg[i], target_vgg[i].detach())

    return loss

pixel_mean_tensor = torch.Tensor((0.485, 0.456, 0.406)).float().reshape(1,3,1,1)
pixel_std_tensor = torch.Tensor((0.229, 0.224, 0.225)).float().reshape(1,3,1,1)

def MSSSIM_loss(real_imgs, fake_imgs):
    B, C, H, W = real_imgs.size()
    pixel_mean = pixel_mean_tensor.repeat(B, 1, H, W).cuda()
    pixel_std = pixel_std_tensor.repeat(B, 1, H, W).cuda()
    scale_back_real_imgs = real_imgs.clone()
    scale_back_real_imgs *= pixel_std
    scale_back_real_imgs += pixel_mean
    scale_back_fake_imgs = fake_imgs.clone()
    scale_back_fake_imgs *= pixel_std
    scale_back_fake_imgs += pixel_mean

    loss = 1 - mssim_calculator(scale_back_fake_imgs, scale_back_real_imgs)
    return loss
