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

def load_model(model, test_model_epoch, model_type):
    model_path = os.path.join(cfg.model_dir, model_type+'_' + str(test_model_epoch) + '.pth.tar')
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['network'])
    return model

def load_optimizer(optimizer, test_model_epoch, model_type):
    opt_path = os.path.join(cfg.model_dir, model_type+'_' + str(test_model_epoch) + '.pth.tar')
    ckpt = torch.load(opt_path)
    optimizer.load_state_dict(ckpt['optimizer'])
    return optimizer

import glob
def load_last_model(model, model_type):
    model_file_list = glob.glob(os.path.join(cfg.model_dir, model_type+'*'))
    last_epoch = max([int(file_name[file_name.find(model_type+'_') + 2: file_name.find('.pth.tar')]) for file_name in model_file_list])
    model = load_model(model, last_epoch, model_type)
    return model

def load_last_optimizer(optimizer, model_type):
    model_file_list = glob.glob(os.path.join(cfg.model_dir, model_type+'*'))
    last_epoch = max([int(file_name[file_name.find(model_type+ '_') + 2: file_name.find('.pth.tar')]) for file_name in model_file_list])
    optimizer = load_optimizer(optimizer, last_epoch, model_type)
    return optimizer

def get_optimizer(model, mode, model_type):
    optm = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    if mode == 'train':
        if cfg.continue_train:
            load_last_optimizer(optm, model_type)
    return optm