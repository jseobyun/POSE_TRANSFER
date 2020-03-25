import numpy as np
import argparse

import torch
from pose_detector.lib.datasets.preprocessing import rtpose_preprocess
from pose_detector.lib.network import im_transform
from pose_detector.lib.config import pose_cfg, update_config


# parser = argparse.ArgumentParser()
# parser.add_argument('--cfg', help='experiment configure file name',
#                     default='/home/vilab/Desktop/yjs/codes/pytorch_Realtime_Multi-Person_Pose_Estimation/experiments/vgg19_368x368_sgd.yaml', type=str)
# parser.add_argument('--weight', type=str,
#                     default='../ckpts/openpose.pth')
# parser.add_argument('opts',
#                     help="Modify config options using the command-line",
#                     default=None,
#                     nargs=argparse.REMAINDER)
# args = parser.parse_args()
#
# # update config file
# update_config(cfg, args)


'''
MS COCO annotation order:
0: nose	   		1: l eye		2: r eye	3: l ear	4: r ear
5: l shoulder	6: r shoulder	7: l elbow	8: r elbow
9: l wrist		10: r wrist		11: l hip	12: r hip	13: l knee
14: r knee		15: l ankle		16: r ankle

The order in this work:
(0-'nose'	1-'neck' 2-'right_shoulder' 3-'right_elbow' 4-'right_wrist'
5-'left_shoulder' 6-'left_elbow'	    7-'left_wrist'  8-'right_hip'
9-'right_knee'	 10-'right_ankle'	11-'left_hip'   12-'left_knee'
13-'left_ankle'	 14-'right_eye'	    15-'left_eye'   16-'right_ear'
17-'left_ear' )
'''

ORDER_COCO = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]

def get_hm_paf(img, model):
    """Computes the averaged heatmap and paf for the given image
    :param multiplier:
    :param origImg: numpy array, the image being processed
    :param model: pytorch model
    :returns: numpy arrays, the averaged paf and heatmap
    """
    inp_size = pose_cfg.DATASET.IMAGE_SIZE

    # padding
    im_croped, im_scale, real_shape = im_transform.crop_with_factor(
        img, inp_size, factor=pose_cfg.MODEL.DOWNSAMPLE, is_ceil=True)


    im_data = rtpose_preprocess(im_croped)

    batch_images= np.expand_dims(im_data, 0)

    # several scales as a batch
    batch_var = torch.from_numpy(batch_images).cuda().float()
    predicted_outputs, _ = model(batch_var)
    output1, output2 = predicted_outputs[-2], predicted_outputs[-1]
    heatmap = output2.cpu().data.numpy().transpose(0, 2, 3, 1)[0]
    paf = output1.cpu().data.numpy().transpose(0, 2, 3, 1)[0]

    return paf, heatmap, im_scale

