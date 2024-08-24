# -*- coding: utf-8 -*-
# @Time    : 2024/8/24 21:12
# @Author  : Lee
# @Project ：kolektor-seg 
# @File    : dataset.py

import cv2
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SegDataset(Dataset):
    def __init__(self,data_root):
        super().__init__()
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

        self.transform = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = self.image_paths[index]
        image = cv2.imdecode(np.fromfile(image_path),cv2.IMREAD_COLOR)
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label,cv2.COLOR_BGR2GRAY)
        # 统一数据集尺寸
        image_resize = cv2.resize(image,(500,1200))
        label_resize = cv2.resize(label,(500,1200))
        # 在线增强
        image_resize = Image.fromarray(image_resize)
        label_resize = Image.fromarray(label_resize)
        image_tensor = self.transform(image_resize)
        label_tensor = self.transform(label_resize)
        return image_tensor,label_tensor

if __name__ == '__main__':
    dataset = SegDataset("../kolektor_aug")
    dataset[0]