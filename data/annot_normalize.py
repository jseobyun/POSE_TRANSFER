import os
import numpy as np
from main.config import cfg
from pose_detector.lib.utils.common import CocoPart


training_annot_path = os.path.join(cfg.source_dir, 'annotations', 'train_annotations.npy')
test_annot_path = os.path.join(cfg.source_dir, 'annotations', 'test_annotations.npy')

R_ankle_idx = CocoPart.RAnkle.value
L_ankle_idx = CocoPart.LAnkle.value
nose_idx = CocoPart.Nose.value

def get_translation(y, s, t):
    s_close = s['close']
    s_far = s['far']
    t_close = t['close']
    t_far = t['far']

    trans = t_far + (y -s_far)/(s_close -s_far) * (t_close-t_far)
    return trans

def get_close_far_point(y_set):

    pt_close = np.max(y_set)
    pt_close_idx = np.argmax(y_set)

    pt_median = np.sort(y_set)[len(y_set)//2]

    distance = np.abs(y_set - pt_median)
    threshold = 0.7 * np.abs(pt_close - pt_median)
    idx1 = distance < threshold
    idx2 = y_set < pt_median
    idx = idx1 * idx2

    selected_y_set = y_set[idx]
    pt_far = np.max(selected_y_set)
    pt_far_idx = np.argwhere(y_set == pt_far)
    pt_set = {'close': pt_close, 'far': pt_far, 'idx_close': pt_close_idx, 'idx_far':pt_far_idx}
    return pt_set

def get_height_ratio(train_nose_set, train_y_set, test_nose_set, test_y_set, s, t):

    idx_s_far = s['idx_far']
    idx_t_far = t['idx_far']

    h_s_close = np.max(np.abs(test_nose_set - test_y_set))
    h_s_far = np.max(np.abs(test_nose_set[idx_s_far] - test_y_set[idx_s_far]))

    h_t_close = np.max(np.abs(train_nose_set - train_y_set))
    h_t_far = np.max(np.abs(train_nose_set[idx_t_far] - train_y_set[idx_t_far]))

    c_close = h_t_close/h_s_close
    c_far = h_t_far/h_s_far
    c = {'close': c_close, 'far': c_far}

    return c

def get_scale(y_set, s, c):
    scale = c['far'] + (y_set-s['far'])/(s['close']-s['far']) * (c['close']-c['far'])
    return scale

if __name__ == "__main__":

    training_annot = np.load(training_annot_path)
    training_fram_num = training_annot[:, 0]
    training_annot = training_annot[:,1:].reshape(-1,18,2) # index 0 has frame number

    test_annot = np.load(test_annot_path)
    test_fram_num = test_annot[:, 0]
    test_annot = test_annot[:, 1:].reshape(-1, 18, 2)  # index 0 has frame number

    # nose and ankle set
    training_nose_y_set = training_annot[:,nose_idx,1]
    training_ankle_y_set = (training_annot[:, R_ankle_idx, 1] + training_annot[:, L_ankle_idx, 1])/2
    test_nose_y_set = test_annot[:, nose_idx, 1]
    test_ankle_x_set = (test_annot[:, R_ankle_idx, 0] + test_annot[:, L_ankle_idx, 0])/2
    test_ankle_y_set = (test_annot[:, R_ankle_idx, 1] + test_annot[:, L_ankle_idx, 1])/2

    # get translation
    t = get_close_far_point(training_ankle_y_set)
    s = get_close_far_point(test_ankle_y_set)

    trans = get_translation(test_ankle_y_set, s, t)

    # get scale
    c = get_height_ratio(training_nose_y_set, training_ankle_y_set, test_nose_y_set, test_ankle_y_set, s, t)
    scale = get_scale(test_ankle_y_set, s, c)

    # get norm test annot
    norm_test_annot = test_annot.copy()
    norm_test_annot[:, :, 0] -= test_ankle_x_set.reshape(-1, 1)
    norm_test_annot[:, :, 1] -= test_ankle_y_set.reshape(-1, 1)
    buf = norm_test_annot[0, 0, :2].copy()
    norm_test_annot[:, :, 0] *= scale.reshape(-1, 1)
    norm_test_annot[:, :, 1] *= scale.reshape(-1, 1)
    buf2 = norm_test_annot[0, 0, :2].copy()
    norm_test_annot[:, :, 0] += test_ankle_x_set.reshape(-1, 1)
    norm_test_annot[:, :, 1] += trans.reshape(-1, 1)
    norm_test_annot[norm_test_annot < 0] = 0

    np.save(os.path.join(cfg.source_dir, 'annotations',  'norm_test_annotations.npy'), norm_test_annot)