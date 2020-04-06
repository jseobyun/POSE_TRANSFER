import cv2
import numpy as np
from enum import Enum

class CocoPart(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18

class Human:
    """
    body_parts: list of BodyPart
    """
    __slots__ = ('body_parts', 'pairs', 'uidx_list', 'score')

    def __init__(self, pairs):
        self.pairs = []
        self.uidx_list = set()
        self.body_parts = {}
        for pair in pairs:
            self.add_pair(pair)
        self.score = 0.0

    @staticmethod
    def _get_uidx(part_idx, idx):
        return '%d-%d' % (part_idx, idx)

    def add_pair(self, pair):
        self.pairs.append(pair)
        self.body_parts[pair.part_idx1] = BodyPart(Human._get_uidx(pair.part_idx1, pair.idx1),
                                                   pair.part_idx1,
                                                   pair.coord1[0], pair.coord1[1], pair.score)
        self.body_parts[pair.part_idx2] = BodyPart(Human._get_uidx(pair.part_idx2, pair.idx2),
                                                   pair.part_idx2,
                                                   pair.coord2[0], pair.coord2[1], pair.score)
        self.uidx_list.add(Human._get_uidx(pair.part_idx1, pair.idx1))
        self.uidx_list.add(Human._get_uidx(pair.part_idx2, pair.idx2))

    def is_connected(self, other):
        return len(self.uidx_list & other.uidx_list) > 0

    def merge(self, other):
        for pair in other.pairs:
            self.add_pair(pair)

    def part_count(self):
        return len(self.body_parts.keys())

    def get_max_score(self):
        return max([x.score for _, x in self.body_parts.items()])

    def __str__(self):
        return ' '.join([str(x) for x in self.body_parts.values()])

    def __repr__(self):
        return self.__str__()

def draw_humans(npimg, humans):
    if np.shape(humans)[0] == 0 :
        return npimg
    image_h, image_w = npimg.shape[:2]
    humans = np.float32(humans)
    centers = {}
    for hidx in range(np.shape(humans)[0]):
        human = humans[hidx]
        # draw point
        for jidx in range(18):
            center = (human[jidx, 0], human[jidx, 1])
            if center[0] == -1 or center[1] == -1:
                continue
            centers[jidx] = center
            cv2.circle(npimg, center, 3, CocoColors[jidx], thickness=3, lineType=8, shift=0)

            # draw line
        for pair_order, pair in enumerate(CocoPairsRender):
            if pair[0] not in centers.keys() or pair[1] not in centers.keys():
                continue
            cv2.line(npimg, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3)
        centers = {}

    return npimg

def draw_bboxes(npimg, bboxes):
    if np.shape(bboxes)[0] == 0:
        return npimg
    for bidx in range(np.shape(bboxes)[0]):
        x, y, w, h = int(bboxes[bidx][0]), int(bboxes[bidx][1]), int(bboxes[bidx][2]), int(bboxes[bidx][3])
        color = (0, 0, 255)
        cv2.line(npimg, (x, y), (x + w, y), color, 3)
        cv2.line(npimg, (x, y), (x, y + h), color, 3)
        cv2.line(npimg, (x+w, y+h), (x, y + h), color, 3)
        cv2.line(npimg, (x + w, y + h), (x+w, y), color, 3)
    return npimg

def humans2array(npimg, humans):
    image_h, image_w = npimg.shape[:2]
    array_humans = []
    for human in humans:
        array_human = []
        for i in range(18):
            if i not in human.body_parts.keys():
                array_human.append([-1, -1])
                continue
            body_part = human.body_parts[i]
            center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
            array_human.append(center)
        array_humans.append(np.array(array_human))
    array_humans = np.array([np.array(x) for x in array_humans])
    return array_humans

def get_bboxes(frame, humans):
    h, w, c = frame.shape
    scale = 1.2
    num_humans = np.shape(humans)[0]
    if num_humans == 0:
        return np.zeros([0])

    bboxes = []
    for human_idx in range(np.shape(humans)[0]):
        xs, ys, centers = [], [], {}
        for joint_idx in range(18):
            center_x = humans[human_idx][joint_idx][0]
            center_y = humans[human_idx][joint_idx][1]
            if center_x == -1 and center_y == -1:
                continue
            center = (int(center_x), int(center_y))
            centers[joint_idx] = center

            xs.append(center[0])
            ys.append(center[1])

        tl_x = min(xs)
        tl_y = min(ys)
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)

        # scale bbox size
        c_x = tl_x + width/2
        c_y = tl_y + height/2
        width *= scale
        height *= scale

        tl_x = c_x - width/2
        tl_y = c_y - height/2

        tl_x = max(0, tl_x)
        tl_y = max(0, tl_y)

        width = min(width , w - tl_x -1)
        height = min(height, h - tl_y -1)

        bboxes.append([tl_x, tl_y, width, height])

    bboxes = np.asarray(bboxes)
    return bboxes

class BodyPart:
    """
    part_idx : part index(eg. 0 for nose)
    x, y: coordinate of body part
    score : confidence score
    """
    __slots__ = ('uidx', 'part_idx', 'x', 'y', 'score')

    def __init__(self, uidx, part_idx, x, y, score):
        self.uidx = uidx
        self.part_idx = part_idx
        self.x, self.y = x, y
        self.score = score

    def get_part_name(self):
        return CocoPart(self.part_idx)

    def __str__(self):
        return 'BodyPart:%d-(%.2f, %.2f) score=%.2f' % (self.part_idx, self.x, self.y, self.score)

    def __repr__(self):
        return self.__str__()
        
CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
              
CocoPairs = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
    (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)
]   # = 19
CocoPairsRender = CocoPairs[:-2]            
              
              
              
