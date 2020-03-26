import os
import cv2
import numpy as np
import torch
from main.config import cfg
from main.model import get_model
from data.dance_dataset import DanceDataset
from torch.utils.data import DataLoader
from utils.train_utils import load_model
from utils.log_utils import global_logger




model = get_model(mode = 'test')
model.eval()

Dance_dataset = DanceDataset(mode='test')
Dance_dataloader =DataLoader(dataset= Dance_dataset, batch_size = cfg.test_batch_size, shuffle = True, num_workers = cfg.num_thread)

for i, (heatmaps, bg_imgs) in enumerate(Dance_dataloader):

    heatmaps, bg_imgs = heatmaps.cuda(), bg_imgs.cuda()
    input = torch.cat([heatmaps, bg_imgs], dim= 1)

    output = model(input, bg_imgs)
    output = output[0].permute(1,2,0)
    output = output.detach().cpu().numpy()
    output = (output * cfg.pixel_std) + cfg.pixel_mean
    clip_idx = output > 1.
    output[clip_idx] == 1
    output = np.array(output*255, dtype=np.uint8)
    cv2.imwrite(os.path.join(cfg.vis_dir, f'{i}.png'), output)

    global_logger.info(f"{i}/{Dance_dataset.len} img done")

    if i > 30:
        break










