import os
import cv2
import numpy as np
from utils.log_utils import global_logger
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms
from main.config import cfg

def put_joint_gaussian_map(center, sigma=cfg.sigma):
    grid_y, grid_x = cfg.input_shape[1], cfg.input_shape[0]

    y_range = np.arange(0, int(grid_y), 1)
    x_range = np.arange(0, int(grid_x), 1)

    yy, xx = np.meshgrid(y_range, x_range)

    d2 = (xx - center[1]) ** 2 + (yy - center[0]) ** 2
    exponent = d2 / 2.0 / sigma / sigma
    mask = exponent <= 4.6052
    mask = mask
    cofid_map = np.exp(-exponent)
    # cofid_map = cofid_map / (sigma*np.sqrt(2*np.pi))
    cofid_map = mask * cofid_map
    # cofid_map = cv2.resize(cofid_map, (cfg.output_shape[1], cfg.output_shape[0]))
    return cofid_map

def get_joint_heatmap(target):
    J, C = target.shape
    heatmaps = np.zeros([J, cfg.input_shape[0], cfg.input_shape[1]] , dtype = np.float32)
    for jidx in range(J):
        joint_xy = target[jidx, 0:2]
        heatmaps[jidx, :, :] = put_joint_gaussian_map(joint_xy)
    return heatmaps

class DanceDataset(Dataset):
    def __init__(self, mode):
        self.logger = global_logger
        if mode == 'train':
            self.data = np.load(os.path.join(cfg.source_dir, 'annotations', 'train_annotations.npy'))
        elif mode =='test':
            self.data = np.load(os.path.join(cfg.source_dir, 'annotations', 'norm_test_annotations.npy'))
        else:
            assert 0, self.logger.info("Invalid dataset type. Choose the mode between 'train' and 'test'.")

        self.img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])

        self.img_idx= np.array(self.data[:, 0, 0], dtype=np.int)
        self.joints = self.data[:, 0, 1:]
        self.len = len(self.data)

        self.bg_img = cv2.imread(os.path.join(cfg.source_dir, mode+'_bg_img.png'))
        self.bg_img = self.img_transform(self.bg_img)

        if mode =='train':
            self.do_augment = cfg.do_augment
        else:
            self.do_augment = False

    def __getitem__(self, index):

        img_name = self.img_idx[index]
        img_name = str(img_name)+'.png'
        img = cv2.imread(os.path.join(cfg.source_dir, 'images', img_name))
        if not isinstance(img, np.ndarray):
            raise IOError(f"Fail to read {img_name}")

        joint = self.joints[index, :].reshape(18,2)
        heatmap = get_joint_heatmap(joint)

        img = self.img_transform(img)

        return img, heatmap, self.bg_img

    def __len__(self):
        return self.len






