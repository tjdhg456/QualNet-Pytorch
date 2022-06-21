#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: lfw.py.py
@time: 2018/12/22 10:00
@desc: lfw dataset loader
'''

import numpy as np
import cv2
import os
import torch.utils.data as data

import torch
import torchvision.transforms as transforms
import random

def img_loader(path, down_size):
    try:
        with open(path, 'rb') as f:
            high_img = cv2.imread(path)
            if len(high_img.shape) == 2:
                high_img = np.stack([high_img] * 3, 2)
            high_img = cv2.resize(high_img, dsize=(128, 128), interpolation=cv2.INTER_NEAREST)
                
            
            if down_size != 128:
                down_img = cv2.resize(high_img, dsize=(down_size, down_size), interpolation=cv2.INTER_NEAREST)
                down_img = cv2.resize(down_img, dsize=(128, 128), interpolation=cv2.INTER_NEAREST)
            else:
                down_img = high_img
            return high_img, down_img
        
    except IOError:
        print('Cannot load image ' + path)

class LFW(data.Dataset):
    def __init__(self, root, file_list, down_size, transform=None, loader=img_loader):
        self.root = root
        self.file_list = file_list
        self.transform = transform
        self.loader = loader
        self.nameLs = []
        self.nameRs = []
        self.folds = []
        self.flags = []
        self.down_size = down_size

        with open(file_list) as f:
            pairs = f.read().splitlines()[1:]
        for i, p in enumerate(pairs):
            p = p.split(' ')
            nameL = p[0]
            nameR = p[1]
            flag = int(p[2])
            fold = i // 600
            self.nameLs.append(nameL)
            self.nameRs.append(nameR)
            self.folds.append(fold)
            self.flags.append(flag)

    def __getitem__(self, index):
        if self.down_size == 0:
            down_size = 128
        else:
            down_size = self.down_size
            
        _, LR_img_l = self.loader(os.path.join(self.root, self.nameLs[index]), down_size)
        _, LR_img_r = self.loader(os.path.join(self.root, self.nameRs[index]), down_size)
        LR_imglist = [LR_img_l, cv2.flip(LR_img_l, 1), LR_img_r, cv2.flip(LR_img_r, 1)]

        if self.transform is not None:
            for i in range(len(LR_imglist)):
                LR_imglist[i] = self.transform(LR_imglist[i])

            imgs = LR_imglist
            return imgs
        else:
            imgs = [torch.from_numpy(i) for i in LR_imglist]
            return imgs

    def __len__(self):
        return len(self.nameLs)

