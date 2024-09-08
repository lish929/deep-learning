# -*- coding: utf-8 -*-
# @Time    : 2024/9/6 11:17
# @Author  : Lee
# @Project ：license-plate 
# @File    : test.py

import cv2
import numpy as np
import os
from ultralytics import YOLO

root = r"D:\datasets\CCPD\images"

# def deal_data(file_name,file_path,images_category_path,labels_category_path,crops_category_path):
for category in os.scandir(root):
    # for category in os.scandir(category.path):
    for file in os.scandir(category.path):
        file_path = file.path
        file_name = file.name
        try:
            image = cv2.imdecode(np.fromfile(file_path),cv2.IMREAD_COLOR)
            height,width,_ = image.shape
            # print(height)
            # print(width)
            if "ccpd_np" not in file_path:
                x1,y1 = int(file_name.split("-")[2].split("_")[0].split("&")[0]),int(file_name.split("-")[2].split("_")[0].split("&")[1])
                x2,y2 = int(file_name.split("-")[2].split("_")[1].split("&")[0]),int(file_name.split("-")[2].split("_")[1].split("&")[1])
                print(x1,y1,x2,y2)
                # cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,0))
                # cv2.imshow("image",image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # exit()

                h,w = (y2-y1),(x2-x1)
                xc,yc = (x1+w/2)/width,(y1+h/2)/height
                h,w = h/height,w/width
                label_path = os.path.join(r"D:\datasets\CCPD\labels",category.name,file_name[0:-4]+".txt")
                label = open(label_path,"a",encoding="utf8")
                label.write("{} {} {} {} {}".format(0,xc,yc,w,h))
                label.close()
            else:
                label_path = os.path.join(r"D:\datasets\CCPD\labels", category.name, file_name[0:-4] + ".txt")
                label = open(label_path, "a", encoding="utf8")
        except Exception as e:
            print(e)
