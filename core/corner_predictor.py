'''
written by Mao-Huan, Hsu
started at 2022/4/15
'''
import cv2
import numpy as np
import os.path as osp

import torch

class CornerPredictor():
    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_file = osp.join(r'core\models', model_name, 'model.pth')
        self.model = torch.load(model_file).to(self.device)
        
        self.model.eval()
        print(f'<{model_name}> model loaded.')

        self.w, self.h = (128, 64)
        self.ow, self.oh = (100, 50)

    def loadImage(self, image, crop=False, pts=None):
        if crop:
            pts = pts.copy().reshape(2, 2)
            offset_x, offset_y = int(0.2*(pts[1, 0]-pts[0, 0])), int(0.2*(pts[1, 1]-pts[0, 1]))
            pts[0, 0], pts[0, 1] = pts[0, 0]-offset_x, pts[0, 1]-offset_y
            pts[1, 0], pts[1, 1] = pts[1, 0]+offset_x, pts[1, 1]+offset_y

            x1_pad, x2_pad, y1_pad, y2_pad = 0, 0, 0, 0
            for i, [x, y] in enumerate(pts):
                if x < 0:
                    x1_pad = 0-x
                    x = 0 
                elif x >= image.shape[1]:
                    x2_pad = x-image.shape[1]+1
                    x = image.shape[1]-1           
                if y < 0:
                    y1_pad = 0-y
                    y = 0
                elif y >= image.shape[0]:
                    y2_pad = y-image.shape[0]+1
                    y = image.shape[0]-1
                pts[i] = [x, y]

            image = image[pts[0, 1]:pts[1, 1]+1, pts[0, 0]:pts[1, 0]+1]
            image = cv2.copyMakeBorder(image, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT, (0, 0, 0))

        self.image = cv2.resize(image, (self.w, self.h))

        return self.image

    def runPrediction(self):
        image = np.array([self.image]).astype('float32')/255
        image = torch.Tensor(image.transpose(0, 3, 1, 2)).to(self.device)
        with torch.no_grad():
            pred = self.model(image)

        pts = self.restoreValues(pred.cpu().numpy().copy())
        res = self.perspective(self.image, pts)

        return res.astype('uint8'), pts.astype('int')

    def restoreValues(self, pts):
        pts = pts.reshape(4, 2)
        pts[:, 0] = np.round(pts[:, 0]*self.w)
        pts[:, 1] = np.round(pts[:, 1]*self.h)

        for i in range(4):
            if pts[:, 0][i] < 0: pts[:, 0][i] = 0
            elif pts[:, 0][i] >= self.w: pts[:, 0][i] = self.w-1
            if pts[:, 1][i] < 0: pts[:, 1][i] = 0
            elif pts[:, 1][i] >= self.h: pts[:, 1][i] = self.h-1

        return pts

    def perspective(self, image, pts):
        res = np.array([[0, 0], [self.ow, 0], [self.ow, self.oh], [0, self.oh]], dtype=np.float32)
        m = cv2.getPerspectiveTransform(pts, res)
        res = cv2.warpPerspective(image, m, (self.ow, self.oh))

        return res
