# -*- coding: utf-8 -*-
# @Time    : 2024/8/24 19:37
# @Author  : Lee
# @Project ：defect-detection 
# @File    : test.py

import torch

from model.deeplabv3 import DeepLabV3

model = DeepLabV3(num_classes=2)
print(model)

x = torch.randn(2,3,224,224)
result = model(x)
print(result.shape)