import torch
from main.config import cfg
from main.model import get_model, get_refiner
from data.dance_dataset import DanceDataset
from torch.utils.data import DataLoader
from utils.log_utils import global_logger
from utils.train_utils import save_model, get_optimizer
from main.loss import L1_loss, MSSSIM_loss

model = get_model(mode = 'test')
model.eval()
refiner = get_refiner(mode= 'train')
Dance_dataset = DanceDataset(mode='train')
Dance_dataloader =DataLoader(dataset= Dance_dataset, batch_size = cfg.batch_size, shuffle = True, num_workers = cfg.num_thread)
R_optimizer = get_optimizer(refiner, mode= 'train', model_type = 'R')



for epoch in range(cfg.start_epoch, cfg.num_epoch):
    for i, (imgs, label_maps, bg_imgs) in enumerate(Dance_dataloader):
        real_imgs, label_maps, bg_imgs = imgs.cuda(), label_maps.cuda(), bg_imgs.cuda()
        input = torch.cat([label_maps, bg_imgs], dim= 1)

        _, fake_imgs = model(input)
        refined_imgs = refiner(fake_imgs)

        loss_L1 = L1_loss(real_imgs, fake_imgs)
        loss_MSSSIM = MSSSIM_loss(real_imgs, fake_imgs)
        R_loss = loss_MSSSIM + loss_L1

        R_optimizer.zero_grad()
        R_loss.backward()
        R_optimizer.step()

        global_logger.info(f"{epoch}/{cfg.num_epoch} epoch, iter: {i}/{int(Dance_dataset.__len__()/cfg.batch_size)}, R_loss: {R_loss.detach().item()}")

    R_state = {'epoch': epoch, 'network': refiner.state_dict(), 'optimizer': R_optimizer.state_dict() }
    save_model(R_state, epoch, model_type='R')


