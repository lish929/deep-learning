# -*- coding: utf-8 -*-
# @Time    : 2024/9/8 19:08
# @Author  : Lee
# @Project ：yolov1 
# @File    : model.py


import torch
from torch import nn

class BasicConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self,x):
        return self.act(self.bn(self.conv(x)))

class CombConv(nn.Module):
    def __init__(self,in_channels,out_channels,repeat_num):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.repeat_num = repeat_num

        self.layers = self._make_layer()

    def _make_layer(self):
        layers = []
        for i in range(self.repeat_num):
            layers.append(BasicConv(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1,padding=0))
            layers.append(BasicConv(in_channels=self.out_channels, out_channels=self.out_channels*2, kernel_size=3, stride=1,padding=1)
)
        return nn.Sequential(*layers)

    def forward(self,x):
        return self.layers(x)


class YOLOv1(nn.Module):
    def __init__(self,):
        super().__init__()
        self.backbone = nn.Sequential(
            BasicConv(3, 64, 7, 2, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BasicConv(64, 192, 3, 1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            CombConv(in_channels=192,out_channels=128,repeat_num=1),
            CombConv(in_channels=256, out_channels=256, repeat_num=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            CombConv(in_channels=512, out_channels=256, repeat_num=4),
            CombConv(in_channels=512, out_channels=512, repeat_num=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            CombConv(in_channels=1024, out_channels=512, repeat_num=2),
            BasicConv(in_channels=1024,out_channels=1024,kernel_size=3,stride=1,padding=1),
            BasicConv(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),

            BasicConv(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            BasicConv(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
        )

    def forward(self,x,s=7,b=2,c=20):
        result_backbone = self.backbone(x)
        n,channel,h,w = result_backbone.shape
        result_temp = result_backbone.view(n,-1)
        result_linear = nn.Linear(channel*w*h,4096)(result_temp)
        result_linear = nn.Linear(4096,s*s*((4+1)*b+c))(result_linear)
        result = result_linear.view(n,s,s,(4+1)*b+c)
        return result

if __name__ == '__main__':
    x = torch.randn(2,3,448,448)
    net = YOLOv1()
    result = net(x)
    print(result.shape)