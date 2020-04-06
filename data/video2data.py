import os
import cv2
import torch
import time
import numpy as np
from utils.log_utils import global_logger
from main.config import cfg
from pose_detector.lib.network.get_models import get_pose_model
from pose_detector.get_results import get_hm_paf
from pose_detector.lib.utils.paf_to_pose import paf_to_pose_cpp
from pose_detector.lib.config import pose_cfg
from pose_detector.lib.utils.common import get_bboxes, humans2array, CocoPart

nose_idx = CocoPart.Nose.value
R_eye_idx = CocoPart.REye.value
L_eye_idx = CocoPart.LEye.value
R_ankle_idx = CocoPart.RAnkle.value
L_ankle_idx = CocoPart.LAnkle.value
others_idx = np.asarray(np.arange(0,18,1))
others_idx = np.delete(others_idx, [nose_idx, R_eye_idx, L_eye_idx])


class BackgroundMaker():
    def __init__(self):
        self.bg_img = np.zeros([cfg.h, cfg.w, 3])
        self.count_map = np.ones([cfg.h, cfg.w, 1]) * 1e-9
        self.max_update = 120
        self.update_num = 0
        self.isDone = False

    def make_bbox_mask(self, bbox):
        binary_mask = np.ones([cfg.h, cfg.w, 3])
        x, y, w, h = int(bbox[0,0]), int(bbox[0,1]), int(bbox[0,2]), int(bbox[0,3])
        binary_mask[y:y+h, x:x+w, :] = 0
        return binary_mask

    def update(self, frame, bbox):
        if self.update_num >= self.max_update:
            self.isDone = True
            return
        bbox_mask = self.make_bbox_mask(bbox)
        tmp_bgimg = frame * bbox_mask
        self.bg_img += tmp_bgimg
        self.count_map += bbox_mask[:,:,0:1]
        self.update_num +=1
        return

    def get_background_img(self):
        if self.isDone:
            output_bg = self.bg_img.copy()

            output_bg[:, :, 0:1] /= self.count_map
            output_bg[:, :, 1:2] /= self.count_map
            output_bg[:, :, 2:] /= self.count_map
            output_bg = np.array(output_bg, dtype=np.uint8)
            return output_bg
        else:
            global_logger.info(f"Background image is not done yet: {self.update_num}/{self.max_update}")
            return None

def check_visibility(humans, thershold = 5):
    # b, 18, 2
    B, J, C = np.shape(humans)

    pick = []
    for bidx in range(B): # always B==1
        isvisible = True
        # check ankle visibility
        R_ankle = humans[bidx, R_ankle_idx, 0]
        L_ankle = humans[bidx, L_ankle_idx, 0]
        if R_ankle == -1 or L_ankle == -1:
            isvisible = False

        # check others visibility
        others = humans[bidx, others_idx, 0]
        if np.shape(others)[0] !=0:
            invalids = np.argwhere(others==-1)
            invalids = np.shape(invalids)[0]
        else:
            invalids = 0

        if invalids >= thershold:
            isvisible = False

        if isvisible:
            pick.append(bidx)

    humans = humans[pick, :, :]
    if np.shape(humans)[0] == 0:
        return np.zeros([0])
    return humans

if __name__ == "__main__":
    mode = 'test' #'dst
    video_name = "test.mp4"
    video_path = os.path.join(cfg.source_dir, video_name)
    vcap = cv2.VideoCapture(video_path)

    detector = get_pose_model()
    bg_maker = BackgroundMaker()

    if vcap.isOpened():
        w = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if w / h != 16 / 9:
            global_logger.warning("Video aspect ratio is not 16(w) : 9(h)")
    else:
        assert 0, global_logger.warning("VideoCapture is not opened")

    frame_num = 0

    annotations = []
    invalids = []
    while True:
        valid, frame = vcap.read()
        if not valid:
            global_logger.warning("VideoCapture doesn't return valid frame.")
            break

        frame_num += 1
        global_logger.info(f"frame_num : {frame_num}")

        frame = cv2.resize(frame, (cfg.w, cfg.h), interpolation = cv2.INTER_CUBIC)
        frame = np.array(frame)

        with torch.no_grad():
            paf, hm, _ = get_hm_paf(frame, detector)

        humans = paf_to_pose_cpp(hm, paf, pose_cfg)
        humans = humans2array(frame, humans)

        if np.shape(humans)[0] == 0:
            invalids.append(frame_num)
            continue

        # check ankle visibility
        humans = check_visibility(humans)

        if np.shape(humans)[0] != 1:
            invalids.append(frame_num)
            continue

        bboxes = get_bboxes(frame, humans)

        frame_label = np.ones([1,1]) * frame_num
        humans = humans.reshape(-1, 36)
        annotation = np.concatenate([frame_label, humans], axis=1)
        annotations.append(annotation)

        if frame_num % 15 == 0:
            bg_maker.update(frame, bboxes)
        if mode == 'train':
            cv2.imwrite(os.path.join(cfg.source_dir, 'images', f'{frame_num}.png'), frame)

    bg_img = bg_maker.get_background_img()
    cv2.imwrite(os.path.join(cfg.source_dir, mode +'_bg_img.png'), bg_img)
    annotations = np.array(annotations).reshape(-1, 37)
    np.save(os.path.join(cfg.source_dir, 'annotations', mode+'_annotations.npy'), annotations)
    print("valid", np.shape(annotations)[0], "invalid", len(invalids), " frame_num", frame_num)

    vcap.release()




