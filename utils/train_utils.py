import os
import torch
from main.config import cfg
from utils.log_utils import global_logger

def update_lr(epoch, optimizer):
    for e in cfg.lr_dec_epoch:
        if epoch < e:
            break
    if epoch < cfg.lr_dec_epoch[-1]:
        idx = cfg.lr_dec_epoch.index(e)
        for g in optimizer.param_groups:
            g['lr'] = cfg.lr / (cfg.lr_dec_factor ** idx)
    else:
        for g in optimizer.param_groups:
            g['lr'] = cfg.lr / (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))

def get_lr(optimizer):
    for g in optimizer.param_groups:
        cur_lr = g['lr']
    return cur_lr

def save_model(state, epoch, model_type):
    save_path = os.path.join(cfg.output_dir, 'model_dump', model_type+f'_{epoch}.pth.tar')
    torch.save(state, save_path)
    global_logger.info(f"{model_type} model is saved in {save_path}")
