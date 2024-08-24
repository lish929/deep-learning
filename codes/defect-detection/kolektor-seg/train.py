# -*- coding: utf-8 -*-
# @Time    : 2024/8/24 22:14
# @Author  : Lee
# @Project ：kolektor-seg 
# @File    : train.py

import argparse
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import sys
import torch
from torch import nn,optim
from torch.utils.data import random_split,DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm

from data.dataset import SegDataset
from model.deeplabv3 import DeepLabV3

class Trainer(object):
    def __init__(self,data_root,log_path,save_path):
        super().__init__()

        # 创建dataloader
        dataset = SegDataset(data_root)
        train_size = int(0.5 * len(dataset))
        val_size = int(0.5 * len(dataset))
        # 划分的长度需要匹配总长度 简化为 0.5：0.5
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size])

        self.train_dataloader = DataLoader(dataset=train_dataset,batch_size=4,shuffle=True,drop_last=True)
        self.val_dataset_dataloader = DataLoader(dataset=val_dataset,batch_size=2,shuffle=True,drop_last=True)

        self.model = DeepLabV3(num_classes=1).to("cuda")
        self.loss = nn.CrossEntropyLoss()
        self.opt = optim.Adam(self.model.parameters())

        self.writer = SummaryWriter(log_path)

        self.sava_path = save_path

    def __call__(self, epoch):
        with tqdm.tqdm(total=epoch) as _tqdm:
            train_loss,train_acc = self._train_epoch()
            val_acc = self._val_epoch()
            _tqdm.set_postfix(train_loss='{:.6f}'.format(train_loss))
            _tqdm.set_postfix(train_acc='{:.6f}'.format(train_acc))
            _tqdm.set_postfix(val_acc_loss='{:.6f}'.format(val_acc))
            _tqdm.update(1)
            self.writer.add_scalar("train_loss",train_loss,epoch)
            self.writer.add_scalar("train_acc",train_acc,epoch)
            self.writer.add_scalar("val_acc",val_acc,epoch)
            torch.save(self.model,os.path.join(self.sava_path,"{}.pt".format(epoch)))

    def _train_epoch(self):
        epoch_loss = 0
        epoch_acc = 0
        flag = 0
        for i,(image,label) in enumerate(self.train_dataloader):
            image,label = image.to("cuda"),label.to("cuda")
            output = self.model(image)
            loss = self.loss(label,output)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            # 可以选择其他的分割评价指标 简单选择acc
            acc = self._compute_acc(label,output)
            epoch_loss = epoch_loss + loss.item()
            epoch_acc = epoch_acc + acc
            flag = i
        return epoch_loss/(flag+1),epoch_acc/(flag+1)

    def _val_epoch(self):
        with torch.no_grad():
            epoch_acc = 0
            flag = 0
            for i,image,label in enumerate(self.train_dataloader):
                image,label = image.to("cuda"),label.to("cuda")
                output = self.model(image)
                # 可以选择其他的分割评价指标 简单选择acc
                acc = self._compute_acc(label,output)
                epoch_acc = epoch_acc + acc
                flag = i
            return epoch_acc/(flag+1)

    def _compute_acc(self,label, output):
        matrix = confusion_matrix(y_true=np.array(label.detach().numpy()).flatten(), y_pred=np.array(output.detach().numpy()).flatten())
        acc = np.diag(matrix).sum() / matrix.sum()
        return acc

def parse_arguments(argv):
    parser = argparse.ArgumentParser()\
    # 可以添加其他配置参数 例如deeplabv3的网络配置等
    parser.add_argument("--data_root",type=str,help="",default="kolektor_aug")
    parser.add_argument("--log_path",type=str,help="",default="weight")
    parser.add_argument("--save_path",type=str,help="",default="log")
    parser.add_argument("--epoch",type=int,help="",default=100)
    return parser.parse_args(argv)

def main(args):
    trainer = Trainer(args.data_root,args.log_path,args.save_path)
    trainer(args.epoch)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))