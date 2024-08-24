# -*- coding: utf-8 -*-
# @Time    : 2024/8/24 20:36
# @Author  : Lee
# @Project ：kolektor-seg 
# @File    : prelabel.py


"""
图像与标注：
    可以利用深度学习 使用预训练模型分批次进行模型迭代，完成标注
    当原始数据标注较为困难或耗时，也可以尝试使用传统方法进行标注，生成标注文件，再利用标注工具进行修改
"""

import cv2
import numpy as np
import os

class PreLabel(object):
    def __init__(self,image_root,label_root):
        super().__init__()
        self.image_root = image_root
        self.label_root = label_root

    def __call__(self, *args, **kwargs):
        for item in os.scandir(self.image_root):
            image = cv2.imdecode(np.fromfile(item.path),cv2.IMREAD_COLOR)
            # canny边缘检测
            # 参数1控制连接 参数2控制起始
            canny_image = cv2.Canny(image,100,180)
            # 连通域分析
            """
            labels_num: 连通域数目
            labels: 图片像素标记（用1，2等表示，每个代表一个连通域）
            status: 每个标记的信息（外接矩阵的x，y，width，height，面积）
            center_ids: 连通域中心点
            """
            labels_num,labels,status,center_ids = cv2.connectedComponentsWithStats(canny_image,connectivity=8)
            for i,(single_status,single_center_id) in enumerate(zip(status,center_ids)):
                # 判断连通域面积
                if single_status[4]<80:
                    labels[np.where(labels==i)] = 0
            labels[np.where(labels>0)] = 255
            labels = np.uint8(labels)
            label_path = os.path.join(self.label_root,item.name)
            cv2.imencode(".jpg",labels)[1].tofile(label_path)

    def _gen_label_file(self):
        pass

if __name__ == '__main__':
    prelabel = PreLabel("../prelabel/images","../prelabel/labels")
    label = prelabel()
    # cv2.imshow("label",label)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()