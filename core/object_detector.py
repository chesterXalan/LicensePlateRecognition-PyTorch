'''
written by Hsu, Mao-Huan 
started at 2021/6/8
'''
import cv2
import numpy as np
import os.path as osp

from core.pytorch_tools.torch_utils import do_detect
from core.pytorch_tools.darknet2pytorch import Darknet

class ObjectDetector():
    def __init__(self, model_name: str):
        config_file = osp.join(r'core\models', model_name, 'model.cfg')
        weights_file = osp.join(r'core\models', model_name, 'darknet.weights')
        names_file = osp.join(r'core\models', model_name, 'classes.names')

        self.net = Darknet(config_file).cuda()
        self.net.load_weights(weights_file)
        self.net.eval()
        self.readNamesFile(names_file)

        print(f'<{model_name}> model loaded.')   

    def readNamesFile(self, nf: str):
        with open(nf, 'r', encoding='utf-8') as f:
            names = f.read().rstrip('\n').split('\n')
        self.names = {i: s for i, s in enumerate(names)}

    def loadImage(self, image: np.ndarray, crop: bool = False, bbox: np.ndarray = None):
        if crop:
            image = image[bbox[0, 1]:bbox[1, 1]+1, bbox[0, 0]:bbox[1, 0]+1] # [y1:y2+1, x1:x2+1] 

        self.height, self.width = image.shape[:2]
        self.image = cv2.resize(image, (self.net.width, self.net.height))

        return image

    def runDetection(self, conf_thrsh: float = 0.6, nms_thrsh: float = 0.4,
                     mode: str = None, multi_res: bool = False, filter_type: str = 'max_conf'):
        bboxes = do_detect(self.net, self.image, conf_thrsh, nms_thrsh)[0]
        if len(bboxes) != 0:
            bboxes = self.restoreValues(bboxes)
            bbox, objs = self.getBboxAndObj(bboxes, mode, multi_res, filter_type)
        else:
            bbox, objs = [], []

        return np.array(bbox).astype('int'), objs

    def restoreValues(self, bboxes: list):
        new_bboxes = []
        for x1, y1, x2, y2, conf, obj in bboxes:
            pts = [[int(x1 * self.width), int(y1 * self.height)], [int(x2 * self.width), int(y2 * self.height)]]
            new_pts = []
            
            for x, y in pts:
                if x < 0: x = 0
                elif x >= self.width: x = self.width - 1
                if y < 0: y = 0
                elif y >= self.height: y = self.height - 1
                new_pts.append([x, y])
            new_bboxes.append([new_pts[0][0], new_pts[0][1], new_pts[1][0], new_pts[1][1], conf, obj])

        return new_bboxes

    def getBboxAndObj(self, bboxes: list, mode: str, multi_res: bool, filter_type: str):
        bbox, objs = [], []

        vehicle_types = ['car', 'bus', 'truck']
        if mode == 'vehicle':
            bbox, objs = self.bboxFilter(bboxes, multi_res, filter_type, vehicle_types)
        elif mode in ['plate', 'number']:
            bbox, objs = self.bboxFilter(bboxes, multi_res, filter_type)
        else:
            print('Unsupported mode.')

        return bbox, objs

    def bboxFilter(self, bboxes: list, multi_res: bool, filter_type: str, obj_filter: list = []):
        bbox, objs = [], []
        max_area = 0
        max_conf = 0.

        for b in bboxes:
            x1, y1, x2, y2, conf, obj = b
            obj = self.names[obj]
            if len(obj_filter) != 0:
                if obj not in obj_filter:
                    continue

            if multi_res:
                bbox.append([(x1, y1), (x2, y2)])
                objs.append(obj)
            else:
                if filter_type == 'max_conf':
                    if conf > max_conf:
                        max_conf = conf 
                        bbox = [(x1, y1), (x2, y2)]
                        objs = obj
                elif filter_type == 'max_area':
                    area = abs((x2-x1) * (y2-y1))
                    if area > max_area:
                        max_area = area 
                        bbox = [(x1, y1), (x2, y2)]
                        objs = obj

        return bbox, objs
