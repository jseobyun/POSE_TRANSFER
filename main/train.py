import os
from tqdm import tqdm
import numpy as np
import torch
from main.config import cfg
from main.model import get_model, get_discriminator, get_Dnet_2D
from data.dance_dataset import DanceDataset
from torch.utils.data import DataLoader
from utils.log_utils import global_logger
from utils.train_utils import update_lr, save_model, get_optimizer
from main.loss import *

model = get_model(mode = 'train')
discriminator = get_discriminator(mode ='train')
Dance_dataset = DanceDataset(mode='train')
Dance_dataloader =DataLoader(dataset= Dance_dataset, batch_size = cfg.batch_size, shuffle = True, num_workers = cfg.num_thread)
G_optimizer = get_optimizer(model, mode= 'train', model_type = 'G')
D_optimizer = get_optimizer(model, mode= 'train', model_type = 'D')


for epoch in range(cfg.start_epoch, cfg.num_epoch):
    for i, (imgs, label_maps, bg_imgs) in enumerate(Dance_dataloader):
        real_imgs, label_maps, bg_imgs = imgs.cuda(), label_maps.cuda(), bg_imgs.cuda()
        input = torch.cat([label_maps, bg_imgs], dim= 1)

        intermediates, fake_imgs = model(input)

        update_lr(epoch, G_optimizer)
        update_lr(epoch, D_optimizer)

        pred_fake = discriminator(torch.cat([label_maps, fake_imgs], dim=1))
        pred_fake_detached = discriminator(torch.cat([label_maps, fake_imgs.detach()], dim=1))
        pred_real = discriminator(torch.cat([label_maps, real_imgs], dim=1))

        loss_G_perceptual = Perceptual_loss(fake_imgs, real_imgs)
        loss_G_MSSSIM = MSSSIM_loss(real_imgs, fake_imgs)
        loss_G_GAN = GAN_loss(pred_fake, 'real')
        loss_G_feat = FM_loss(pred_fake, pred_real)
        loss_G_intermediate = Intermediate_loss(intermediates, real_imgs)
        loss_G_L1 = L1_loss(fake_imgs, real_imgs)
        loss_G_SmoothL1= Smooth_L1_loss(fake_imgs, real_imgs)

        loss_D_fake = GAN_loss(pred_fake_detached, 'fake')
        loss_D_real = GAN_loss(pred_real, 'real')


        G_loss = 4*loss_G_GAN + 2.0*loss_G_feat + loss_G_perceptual + 0.1*(loss_G_L1 + loss_G_SmoothL1) + loss_G_intermediate + 3.0*loss_G_MSSSIM
        D_loss = 0.5*(loss_D_fake + loss_D_real)

        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        global_logger.info(f"{epoch}/{cfg.num_epoch} epoch, iter: {i}/{int(Dance_dataset.__len__()/cfg.batch_size)}, G_loss: {G_loss.detach().item()} , D_loss: {D_loss.detach().item()}")


    G_state = {'epoch': epoch, 'network': model.state_dict(), 'optimizer': G_optimizer.state_dict() }
    D_state = {'epoch': epoch, 'network': discriminator.state_dict(), 'optimizer': D_optimizer.state_dict()}
    save_model(G_state, epoch, model_type='G')
    save_model(D_state, epoch, model_type='D')

