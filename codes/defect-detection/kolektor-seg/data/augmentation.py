# -*- coding: utf-8 -*-
# @Time    : 2024/8/24 21:17
# @Author  : Lee
# @Project ：kolektor-seg 
# @File    : augmentation.py

"""
离线增强
"""

import cv2
import numpy as np
import os
import random

class Augmentation(object):
    def __init__(self,data_root,aug_root):
        super().__init__()
        self.data_root = data_root
        self.aug_root = aug_root

        self.image_paths = []
        self.label_paths = []
        for category in os.scandir(data_root):
            category_path = category.path
            for item in os.scandir(category_path):
                item_name = item.name
                item_path = item.path
                if "label" not in item_name:
                    self.image_paths.append(item_path)
                else:
                    self.label_paths.append(item_path)

    def __call__(self, *args, **kwargs):
        for i,(image_path,label_path) in enumerate(zip(self.image_paths,self.label_paths)):
            image = cv2.imdecode(np.fromfile(image_path), cv2.IMREAD_COLOR)
            # label = cv2.imdecode(np.fromfile(label_path), cv2.IMREAD_COLOR)
            label = cv2.imread(label_path)
            flag = random.randint(0,2)
            if flag == 0:
                image_aug,label_aug = self._change_light_value(image,label)
            elif flag == 1:
                image_aug,label_aug = self._flip_w(image,label)
            else:
                image_aug,label_aug = self._flip_h(image,label)
            if not os.path.exists(os.path.join(self.aug_root,image_path.split("\\")[-2])):
                os.makedirs(os.path.join(self.aug_root,image_path.split("\\")[-2]))
            cv2.imencode(".jpg",image)[1].tofile(os.path.join(self.aug_root,image_path.split("\\")[-2],image_path.split("\\")[-1]))
            cv2.imencode(".bmp",label)[1].tofile(os.path.join(self.aug_root,label_path.split("\\")[-2],label_path.split("\\")[-1]))
            cv2.imencode(".jpg",image_aug)[1].tofile(os.path.join(self.aug_root,image_path.split("\\")[-2],image_path.split("\\")[-1][:-4]+"_aug"+".jpg"))
            cv2.imencode(".bmp",label_aug)[1].tofile(os.path.join(self.aug_root,label_path.split("\\")[-2],label_path.split("\\")[-1][:-4]+"_aug"+".bmp"))

    def _change_light_value(self,image,label):
        image_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        image_hsv[:,:,2] = np.uint8(image_hsv[:,:,2]/3*2)
        image_bgr = cv2.cvtColor(image_hsv,cv2.COLOR_HSV2BGR)
        return image_bgr,label

    def _flip_w(self,image,label):
        image_flip = cv2.flip(image,0)
        label_flip = cv2.flip(label,0)
        return image_flip,label_flip

    def _flip_h(self,image,label):
        image_flip = cv2.flip(image,1)
        label_flip = cv2.flip(label,1)
        return image_flip,label_flip

if __name__ == '__main__':
    augmentation = Augmentation("../kolektor","../kolektor_aug")
    augmentation()