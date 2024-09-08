# -*- coding: utf-8 -*-
# @Time    : 2024/9/6 15:29
# @Author  : Lee
# @Project ：license-plate 
# @File    : split_datas.py
# -*- coding: utf-8 -*-
# @Time    : 2024/9/6 11:34
# @Author  : Lee
# @Project ：license-plate
# @File    : deal_datas.py

"""
处理CCPD 2019以及CCPD 2020数据集
    CCPD 2019: https://drive.usercontent.google.com/download?id=1rdEsCUcIUaYOVRkx5IMTRNA7PcGMmSgc&export=download
    CCPD 2020: https://drive.usercontent.google.com/download?id=1m8w1kFxnCEiqz_-t2vTcgrgqNIv986PR&export=download

04-90_267-158&448_542&553-541&553_162&551_158&448_542&450-0_1_3_24_27_33_30_24-99-116.jpg
00205459770115-90_85-352&516_448&547-444&547_368&549_364&517_440&515-0_0_22_10_26_29_24-128-7.jpg

生成yolo标签
剪裁车牌图片 根据角度矫正图片

"""


import argparse
import os
import sys

def split_datas(root):
    paths = []
    for category in os.scandir(root):
        paths.append([os.path.join(category.path,file_name) for file_name in os.listdir(category.path)])

    train_paths = []
    val_paths = []
    test_paths = []
    for item in paths:
        train_paths.extend(item[0:int(len(item)*0.7)])
        val_paths.extend(item[int(len(item)*0.7):int(len(item)*0.9)])
        test_paths.extend(item[int(len(item)*0.9):])

    train_txt = open(os.path.join(root,"train.txt"),"a",encoding="utf8")
    val_txt = open(os.path.join(root,"val.txt"),"a",encoding="utf8")
    test_txt = open(os.path.join(root,"test.txt"),"a",encoding="utf8")

    for train_path in train_paths:
        train_txt.write(train_path+"\n")
    for val_path in val_paths:
        val_txt.write(val_path+"\n")
    for test_path in test_paths:
        test_txt.write(test_path+"\n")


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",type=str,help="",default=r"D:\datasets\CCPD\images")
    return parser.parse_args(argv)

def main(args):
    split_datas(args.root)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
