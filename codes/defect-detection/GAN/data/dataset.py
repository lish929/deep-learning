# -*- coding: utf-8 -*-
# @Time    : 2024/8/29 10:38
# @Author  : Lee
# @File    : dataset.py
# @Description :


import cv2
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class GanDataset(Dataset):
    def __init__(self,image_root,annotation_root,height,width):
        super().__init__()
        self.annotation_root = annotation_root

        image_paths = []
        annotation_paths = []

        for category in os.scandir(image_root):
            category_path = category.path
            for file in os.scandir(category_path):
                image_paths.append(file.path)

        if annotation_root != "":
            for category in os.scandir(annotation_root):
                category_path = category.path
                for file in os.scandir(category_path):
                    annotation_paths.append(file.path)

        self.image_paths = image_paths
        self.annotation_paths = annotation_paths

        self.transform = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((height,width)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        if self.annotation_root != "":
            annotation_path = self.annotation_paths[index]
            annotation = Image.open(annotation_path)
            return transforms.ToTensor()(image),transforms.ToTensor()(annotation)
        return self.transform(image)

if __name__ == '__main__':
    dataset = GanDataset(r"D:\data\datasets\mvtec_anomaly_detection\bottle\test",r"D:\data\datasets\mvtec_anomaly_detection\bottle\ground_truth")
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][0])
    print(dataset[0][1])