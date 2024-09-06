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
from concurrent.futures import ThreadPoolExecutor,as_completed
import cv2
import numpy as np
import os
import shutil
import sys
from tqdm import tqdm

PROVINCES = [
    "皖", "沪", "津", "渝", "冀",
    "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣",
    "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云",
    "西", "陕", "甘", "青", "宁",
    "新"]

CHARS = [
    "A", "B", "C", "D", "E",
    "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q",
    "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "0",
    "1", "2", "3", "4", "5",
    "6", "7", "8", "9"]


def collect_records(ccpd_path):
    images_path = os.path.join(ccpd_path,"images")
    labels_path = os.path.join(ccpd_path,"labels")
    crops_path = os.path.join(ccpd_path, "crops")

    if not os.path.exists(images_path):
        os.makedirs(images_path)
    if not os.path.exists(labels_path):
        os.makedirs(labels_path)
    if not os.path.exists(crops_path):
        os.makedirs(crops_path)

    records = {}
    for category in os.scandir(ccpd_path):
        category_name = category.name
        category_path = category.path

        images_category_path = os.path.join(ccpd_path, "images",category_name)
        labels_category_path = os.path.join(ccpd_path, "labels",category_name)
        crops_category_path = os.path.join(ccpd_path, "crops",category_name)

        if not os.path.isdir(category_path) or category_name == "splits" or category_name == "images" or category_name == "labels":
            continue
        if not os.path.exists(images_category_path):
            os.makedirs(images_category_path)
        if not os.path.exists(labels_category_path):
            os.makedirs(labels_category_path)
        if not os.path.exists(crops_category_path):
            os.makedirs(crops_category_path)

        for file in os.scandir(category_path):
            file_name = file.name
            file_path = file.path
            records[file_path] = [file_name,file_path,images_category_path,labels_category_path,crops_category_path]
    return records


def deal_data(file_name,file_path,images_category_path,labels_category_path,crops_category_path):
    try:
        image = cv2.imdecode(np.fromfile(file_path),cv2.IMREAD_COLOR)
        height,width,_ = image.shape
        x1,y1 = int(file_name.split("-")[2].split("_")[0].split("&")[0]),int(file_name.split("-")[2].split("_")[0].split("&")[1])
        x2,y2 = int(file_name.split("-")[2].split("_")[1].split("&")[0]),int(file_name.split("-")[2].split("_")[1].split("&")[1])
        h,w = (y2-y1)/height,(x2-x1)/width
        xc,yc = (x1+w/2)/width,(y1+h/2)/height
        label_path = os.path.join(labels_category_path,file_name[0:-4]+".txt")
        label = open(label_path,"a",encoding="utf8")
        label.write("{} {} {} {} {}".format(0,xc,yc,w,h))
        label.close()
        crop_image = image[y1:y2, x1:x2]
        chars = ""
        nums = file_name.split("-")[-3].split("_")
        for i, num in enumerate(nums):
            if i == 0:
                chars.join(PROVINCES[int(num)])
            else:
                chars.join(CHARS[int(num)])
        crop_path = os.path.join(crops_category_path, file_name[0:-4] + "_crop@{}.jpg".format(chars))
        cv2.imencode(".jpg", crop_image)[1].tofile(crop_path)
        shutil.move(file_path, images_category_path)
    except Exception as e:
        print(e)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ccpd_path",type=str,help="",default=r"D:\datasets\CCPD")
    return parser.parse_args(argv)

def main(args):
    records = collect_records(args.ccpd_path)
    tasks = []
    pool = ThreadPoolExecutor(32)

    for file_path in records.keys():
        tasks.append(pool.submit(deal_data,records[file_path][0],records[file_path][1],records[file_path][2],records[file_path][3],records[file_path][4]))
    for task in tqdm(as_completed(tasks),total=len(tasks),desc="deal datas"):
        pass
    for category in os.scandir(args.ccpd_path):
        category_name = category.name
        category_path = category.path
        if category_name == "images" or category_name == "labels" or category_name == "crops":
            continue
        shutil.rmtree(category_path)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
