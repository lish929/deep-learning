# -*- coding: utf-8 -*-
# @Time    : 2024/9/4 21:59
# @Author  : Lee
# @Project ：license-plate 
# @File    : LPRNet.py

import torch
from torch import nn

class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.basic_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels//4,kernel_size=1,stride=1,padding=0),
            nn.Conv2d(in_channels=out_channels//4,out_channels=out_channels//4,kernel_size=1,stride=1,padding=1),
            nn.Conv2d(in_channels=out_channels//4,out_channels=out_channels,kernel_size=1,stride=1,padding=0)
        )

    def forward(self,x):
        return self.basic_block(x)

class SmallBasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.small_basic_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 4, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=out_channels // 4, out_channels=out_channels // 4, kernel_size=(3,1), stride=1,
                      padding=(1,0)),
            nn.Conv2d(in_channels=out_channels // 4, out_channels=out_channels // 4, kernel_size=(1, 3), stride=1,
                      padding=(0, 1)),
            nn.Conv2d(in_channels=out_channels // 4, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        )
    def forward(self,x):
        return self.small_basic_block(x)

class MixedInputBlock(nn.Module):
    def __init__(self,in_channels,out_channels=64,block=BasicBlock):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3,stride=1)
        self.block = block(in_channels=out_channels,out_channels=out_channels*2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3,stride=(2,1))

    def forward(self,x):
        return self.max_pool2(self.block(self.max_pool1(self.conv1(x))))

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,block=BasicBlock):
        super().__init__()
        self.block = block(in_channels=in_channels,out_channels=out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=3,stride=(2,1))

    def forward(self,x):
        return self.max_pool(self.block(x))

class LPRNet(nn.Module):
    def __init__(self,in_channels=3,out_channels=64,class_num=7):
        super().__init__()
        self.input_block = MixedInputBlock(3)
        self.block1 = BasicBlock(in_channels=out_channels*2,out_channels=out_channels*4)
        self.block2 = ConvBlock(in_channels=out_channels*4,out_channels=out_channels*4)
        self.dropout1 = nn.Dropout(0.5)
        self.conv1 = nn.Conv2d(in_channels=out_channels*4,out_channels=out_channels*4, kernel_size=(4,1),stride=1,padding=0)
        self.dropout2 = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(in_channels=out_channels*4,out_channels=class_num,kernel_size=(13,1),stride=1,padding=0)
    def forward(self,x):
        return self.conv2(self.dropout2(self.conv1(self.dropout1(self.block2(self.block1(self.input_block(x)))))))

if __name__ == '__main__':
    x = torch.randn(2,3,24,94)
    # net = BasicBlock(64,64)
    # net = SmallBasicBlock(64,64)
    # net = MixedInputBlock(in_channels=3)
    net = LPRNet(in_channels=3)
    result = net(x)
    print(result.shape)