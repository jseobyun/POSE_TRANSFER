import torch
import numpy as np
import torch.nn as nn
from main.config import cfg
from nets.hg_net import hg_net
from nets.pix2pix_G_net import get_Gnet_2D
from nets.pix2pix_D_net import get_Dnet_2D
from utils.train_utils import load_last_model, load_model





def init_weights(m):
    if isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, std=0.001)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0)


def get_model(mode):
    model = hg_net()
    # model = get_Gnet_2D()
    if mode =='train':
        if cfg.continue_train:
            model = load_last_model(model, model_type= 'G')
        else:
            model.apply(init_weights)

        model.train()
    elif mode =='test':
        model = load_model(model, cfg.test_model_epoch, model_type='G')
        model.eval()
    return model.cuda()

def get_refiner(mode):
    model = get_Gnet_2D()
    if mode =='train':
        if cfg.continue_train:
            model = load_last_model(model, model_type='R')
        else:
            model.apply(init_weights)
        model.train()
    elif mode == 'test':
        model = load_model(model, cfg.refine_model_epoch, model_type='R')
        model.eval()
    return model.cuda()


def get_discriminator(mode):
    model = get_Dnet_2D()#DiscNet()
    if mode == 'train':
        if cfg.continue_train:
            model = load_last_model(model, model_type = 'D')
        if not cfg.continue_train:
            model.apply(init_weights)
        model.train()
    if mode =='test':
        model = load_model(model, cfg.test_model_epoch, model_type= 'D')
        model.eval()
    return model.cuda()