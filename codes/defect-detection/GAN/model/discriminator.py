# -*- coding: utf-8 -*-
# @Time    : 2024/8/29 9:43
# @Author  : Lee
# @File    : discriminator.py
# @Description :


import numpy as np
import torch
from torch import nn

# 将输入图片转换到图片的真假的概率
class Discriminator(nn.Module):
    def __init__(self,channel,height,width):
        super().__init__()
        self.model = nn.Sequential(
            # np.prod 计算数组元素乘积
            nn.Linear(int(np.prod((channel,height,width))),512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self,x):
        # x.shape x.size() 转换输入到[batch,channel*height,width]]
        x_flatten = x.view(x.shape[0],-1)
        result = self.model(x_flatten)
        return result


if __name__ == '__main__':
    model = Discriminator(3,1024,1024)
    x = torch.randn(2,3,1024,1024)
    result = model(x)