# -*- coding: utf-8 -*-
# @Time    : 2024/8/29 9:36
# @Author  : Lee
# @File    : generator.py
# @Description :


import numpy as np
import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self,channel,height,width,latent_dim):
        super().__init__()
        self.channel = channel
        self.height = height
        self.width = width
        def block(in_features,out_features,normalize=True):
            layers = [nn.Linear(in_features,out_features)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_features,0.8))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            return layers

        self.model = nn.Sequential(
            # 为什么选择性的使用normalize？
            *block(latent_dim,128,normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod((channel,height,width)))),
            # 为什么使用Than激活函数（范围-1，1）？
            nn.Tanh()
        )

    def forward(self,z):
        result = self.model(z)
        result = result.view(z.shape[0],self.channel,self.height,self.width)
        return result

if __name__ == '__main__':
    model = Generator(3,1024,1024,100)
    z = torch.randn(2,100)
    print(z)
    result = model(z)
    print(result.shape)