# -*- coding: utf-8 -*-
# @Time    : 2024/8/29 11:21
# @Author  : Lee
# @File    : train.py
# @Description :


import argparse
import numpy as np
import os
import sys
import torch
from torch import nn
from torch import optim
from torch.utils.data import random_split,DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import GanDataset
from model.discriminator import Discriminator
from model.generator import Generator

class Trainer(object):
    def __init__(self,image_root,annotation_root,batch_size,channel,height,width,latent_dim,log_path,save_path):
        super().__init__()
        dataset = GanDataset(image_root,annotation_root,height,width)
        dataset_length = len(dataset)
        train_length = int(dataset_length*0.8)
        val_length = dataset_length - train_length
        train_dataset,val_dataset = random_split(dataset,[train_length,val_length])
        self.train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
        self.val_dataloader = DataLoader(val_dataset,batch_size=batch_size//2,shuffle=True,drop_last=True)

        self.net_D = Discriminator(channel,height,width).cuda()
        self.net_G = Generator(channel,height,width,latent_dim).cuda()
        self.loss = nn.BCELoss()
        self.optim_D = optim.Adam(self.net_D.parameters())
        self.optim_G = optim.Adam(self.net_G.parameters())

        self.writer = SummaryWriter(log_path)

        self.latent_dim = latent_dim
        self.save_path = save_path

    def __call__(self,epoch):
        for i in range(epoch):
            print("epoch:{}/{}".format(i+1,epoch))
            self.train_epoch(i)
            self.val_epoch(i)

    def train_epoch(self,epoch):
        with tqdm(total=len(self.train_dataloader)) as _tqdm:
            _tqdm.set_description("train")
            epoch_loss_g = 0
            epoch_loss_d = 0
            for i,(image) in enumerate(self.train_dataloader):
                image = image.cuda()
                # 潜空间变量
                z = torch.tensor(np.random.normal(0,1,(image.shape[0],self.latent_dim)),dtype=torch.float32).cuda()
                # 判别器比对标准 真
                valid = torch.tensor(np.ones((image.shape[0],1)),dtype=torch.float32).cuda()
                # 判别器比对标准 假
                fake = torch.tensor(np.zeros((image.shape[0],1)),dtype=torch.float32).cuda()

                gen_image = self.net_G(z)
                # 生成器需要让生成的图片尽可能的欺骗过判别器（让判别器将生成图片识别为真）
                loss_g = self.loss(self.net_D(gen_image),valid)
                self.optim_G.zero_grad()
                loss_g.backward()
                self.optim_G.step()

                # 判别器既能识别真实图片 也能识别生成图片
                loss_real = self.loss(self.net_D(image),valid)
                loss_fake = self.loss(self.net_D(gen_image.detach()),fake)
                loss_d = (loss_real+loss_fake)/2
                self.optim_D.zero_grad()
                loss_d.backward()
                self.optim_D.step()

                epoch_loss_g = epoch_loss_g + loss_g.item()
                epoch_loss_d = epoch_loss_d + loss_d.item()

                _tqdm.set_postfix(loss_g="{:.6f}".format(loss_g.item()),loss_d="{:.6f}".format(loss_d.item()))
                _tqdm.update(1)

            self.writer.add_scalar("loss_g",epoch_loss_g/len(self.train_dataloader),epoch)
            self.writer.add_scalar("loss_d",epoch_loss_d/len(self.train_dataloader),epoch)

            if epoch%100 == 0:
                torch.save(self.net_G,os.path.join(self.save_path,"net_G_{}.pt".format(epoch)))
                torch.save(self.net_D,os.path.join(self.save_path,"net_D_{}.pt".format(epoch)))

    def val_epoch(self,epoch):
        with torch.no_grad():
            with tqdm(total=len(self.val_dataloader)) as _tqdm:
                _tqdm.set_description("val")
                epoch_acc_g = 0
                epoch_acc_d = 0
                for i,(image) in enumerate(self.val_dataloader):
                    image = image.cuda()
                    # 潜空间变量
                    z = torch.tensor(np.random.normal(0, 1, (image.shape[0], self.latent_dim)),dtype=torch.float32).cuda()
                    # 判别器比对标准 真
                    valid = torch.tensor(np.ones((image.shape[0], 1)),dtype=torch.float32).cuda()
                    # 判别器比对标准 假
                    fake = torch.tensor(np.zeros((image.shape[0],1)),dtype=torch.float32).cuda()

                    gen_image = self.net_G(z)
                    # 计算生成器的欺骗准确率 也就是生成器根据潜空间变量生成图片所能欺骗判别器的概率
                    acc_g = torch.eq(self.net_D(gen_image),valid).float().mean()

                    # 判别器的判别准确率
                    acc_real = torch.eq(self.net_D(image), valid).float().mean()
                    acc_fake = torch.eq(self.net_D(gen_image), fake).float().mean()
                    acc_d = (acc_real+acc_fake)/2

                    epoch_acc_g = epoch_acc_g + acc_g.item()
                    epoch_acc_d = epoch_acc_d + acc_d.item()

                    _tqdm.set_postfix(acc_g="{:.6f}".format(acc_g.item()),acc_d="{:.6f}".format(acc_d.item()))
                    _tqdm.update(1)

                self.writer.add_scalar("loss_g", epoch_acc_g / len(self.train_dataloader), epoch)
                self.writer.add_scalar("loss_d", epoch_acc_d / len(self.train_dataloader), epoch)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_root",type=str,help="",default=r"D:\data\datasets\mvtec_anomaly_detection\bottle\train")
    parser.add_argument("--annotation_root",type=str,help="",default=r"")
    parser.add_argument("--batch_size",type=int,help="",default=8)
    parser.add_argument("--channel",type=int,help="",default=3)
    parser.add_argument("--height",type=int,help="",default=200)
    parser.add_argument("--width",type=int,help="",default=200)
    parser.add_argument("--latent_dim",type=int,help="",default=100)
    parser.add_argument("--log_path", type=str, help="",
                        default=r"D:\project\GAN\log")
    parser.add_argument("--save_path", type=str, help="",
                        default=r"D:\project\GAN\weight")
    parser.add_argument("--epoch", type=int, help="", default=1000)

    return parser.parse_args(argv)

def main(args):
    trainer = Trainer(args.image_root,args.annotation_root,args.batch_size,args.channel,args.height,args.width,args.latent_dim,args.log_path,args.save_path)
    trainer(args.epoch)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))