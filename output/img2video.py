import os
import matplotlib.pyplot as plt
import numpy as np
import imageio
import glob
from main.config import cfg
from utils.log_utils import global_logger

img_path = os.path.join(cfg.output_dir, 'vis')
writer = imageio.get_writer(os.path.join(cfg.result_dir, 'result.mp4'), fps=cfg.test_fps)

img_file_list = np.array(glob.glob(os.path.join(img_path, '*.png')))
sort_idx = np.array([int(file_name[file_name.find('vis/')+4 : file_name.find('.png')]) for file_name in img_file_list])
sort_idx = np.argsort(sort_idx)
img_file_list = img_file_list[sort_idx]

for i, img_path in enumerate(img_file_list):
    im = plt.imread(img_path)
    im = np.asarray(im*255, dtype=np.uint8)
    writer.append_data(im)
    global_logger.info(f"{i+1}/{len(img_file_list)} img2video done")

writer.close()