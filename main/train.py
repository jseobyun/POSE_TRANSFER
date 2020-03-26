import os
from tqdm import tqdm
import numpy as np
import torch
from main.config import cfg
from main.model import get_model, get_discriminator
from data.dance_dataset import DanceDataset
from torch.utils.data import DataLoader
from utils.log_utils import global_logger
from utils.train_utils import update_lr, save_model, get_optimizer
from main.loss import L1_loss, BCE_loss

model = get_model(mode = 'train')
discriminator = get_discriminator(mode ='train')
Dance_dataset = DanceDataset(mode='train')
Dance_dataloader =DataLoader(dataset= Dance_dataset, batch_size = cfg.batch_size, shuffle = True, num_workers = cfg.num_thread)
G_optimizer = get_optimizer(model, mode= 'train', model_type = 'G')
D_optimizer = get_optimizer(model, mode= 'train', model_type = 'D')

for epoch in range(cfg.start_epoch, cfg.num_epoch):
    for i, (imgs, heatmaps, bg_imgs) in enumerate(Dance_dataloader):

            imgs, heatmaps, bg_imgs = imgs.cuda(), heatmaps.cuda(), bg_imgs.cuda()
            input = torch.cat([heatmaps, bg_imgs], dim= 1)
            output = model(input, bg_imgs)

            update_lr(epoch, G_optimizer)
            update_lr(epoch, D_optimizer)

            # G update
            if i%1 == 0:
                model.train()
                discriminator.eval()

                pred_fake = discriminator(output)
                fake_loss = BCE_loss(pred_fake, 'fake')

                G_loss = L1_loss(imgs, output) + 0.5 * fake_loss
                G_optimizer.zero_grad()
                G_loss.backward()
                G_optimizer.step()

            # D upate
            if i%1 == 0:
                model.eval()
                discriminator.train()

                pred_fake = discriminator(output.detach())
                pred_real = discriminator(imgs)

                D_optimizer.zero_grad()
                fake_loss = BCE_loss(pred_fake, 'fake')
                real_loss = BCE_loss(pred_real, 'real')
                D_loss = 0.5*(fake_loss + real_loss)
                D_loss.backward()
                D_optimizer.step()

            global_logger.info(f"{epoch}/{cfg.num_epoch} epoch, iter: {i}/{int(Dance_dataset.__len__()/cfg.batch_size)}, G_loss: {G_loss.detach().item()} , D_loss: {D_loss.detach().item()}") #


    G_state = {'epoch': epoch, 'network': model.state_dict(), 'optimizer': G_optimizer.state_dict() }
    D_state = {'epoch': epoch, 'network': discriminator.state_dict(), 'optimizer': D_optimizer.state_dict()}
    save_model(G_state, epoch, model_type='G')
    save_model(D_state, epoch, model_type='D')

