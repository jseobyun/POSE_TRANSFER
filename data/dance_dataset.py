import os
import cv2
import math
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

limb_list = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
    [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
    [2, 16], [5, 17]] #19
def create_label_map(joints):
    label_map = np.zeros([19, cfg.input_shape[0], cfg.input_shape[1]], dtype=np.float32)
    buf_img = np.zeros([cfg.input_shape[0], cfg.input_shape[1]], dtype=np.float32)

    for lidx in range(len(limb_list)):
        limb = limb_list[lidx]
        j1, j2 = joints[limb[0], :], joints[limb[1], :]
        if -1 in j1 or -1 in j2 or 0 in j1 or 0 in j2:
            continue
        center = tuple(np.round((j1+j2)/2).astype(int))
        limb_dir = j2 - j1
        limb_length = np.linalg.norm(limb_dir)
        angle = math.degrees(math.atan2(limb_dir[1], limb_dir[0]))
        polygon = cv2.ellipse2Poly(center, (int(limb_length/2), 4), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(label_map[lidx, :, :], polygon, 1)
        cv2.fillConvexPoly(buf_img, polygon, 1)
    return label_map, buf_img



class DanceDataset(Dataset):
    def __init__(self, mode):
        self.mode = mode
        self.logger = global_logger
        if mode == 'train':
            self.data = np.load(os.path.join(cfg.source_dir, 'annotations', 'train_annotations.npy'))
            self.img_idx = np.array(self.data[:, 0], dtype=np.int)
            self.joints = self.data[:, 1:]
        elif mode =='test':
            self.data = np.load(os.path.join(cfg.source_dir, 'annotations', 'norm_test_annotations.npy'))
            self.joints = self.data
        else:
            assert 0, self.logger.info("Invalid dataset type. Choose the mode between 'train' and 'test'.")

        self.img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])
        self.len = len(self.data)

        self.bg_img = cv2.imread(os.path.join(cfg.source_dir, 'train_bg_img.png'))
        self.bg_img = self.img_transform(self.bg_img)

        if mode =='train':
            self.do_augment = cfg.do_augment
        else:
            self.do_augment = False

    def __getitem__(self, index):
        joint = self.joints[index, :].reshape(18, 2)

        # heatmap = get_joint_heatmap(joint)
        label_map, buf_img = create_label_map(joint)

        if self.mode == 'train':
            img_name = self.img_idx[index]
            img_name = str(img_name)+'.png'
            img = cv2.imread(os.path.join(cfg.source_dir, 'images', img_name))
            if not isinstance(img, np.ndarray):
                raise IOError(f"Fail to read {img_name}")
            img = self.img_transform(img)
            return img, label_map, self.bg_img
        elif self.mode == 'test':
            return label_map, self.bg_img
    def __len__(self):
        return self.len






