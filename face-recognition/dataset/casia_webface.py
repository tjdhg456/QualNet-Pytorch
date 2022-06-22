#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: casia_webface.py
@time: 2018/12/21 19:09
@desc: CASIA-WebFace dataset loader
'''

import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import cv2
import os
import torch
import random

def img_loader(path, down_size):
    try:
        with open(path, 'rb') as f:
            high_img = cv2.imread(path)
            if len(high_img.shape) == 2:
                high_img = np.stack([high_img] * 3, 2)
            
            if down_size != 112:
                down_img = cv2.resize(high_img, dsize=(down_size, down_size), interpolation=cv2.INTER_NEAREST)
                down_img = cv2.resize(down_img, dsize=(112, 112), interpolation=cv2.INTER_NEAREST)
            else:
                down_img = high_img
            return high_img, down_img
        
    except IOError:
        print('Cannot load image ' + path)
        

class CASIAWebFace(data.Dataset):
    def __init__(self, root, file_list, down_size, transform=None, loader=img_loader, flip=True):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.down_size = down_size
        
        image_list = []
        label_list = []
        with open(file_list) as f:
            img_label_list = f.read().splitlines()

        for info in img_label_list:
            image_path, label_name = info.split('  ')
            image_list.append(image_path)
            label_list.append(int(label_name))

        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))

        self.flip = flip
        print("dataset size: ", len(self.image_list), '/', self.class_nums)

    def __getitem__(self, index):
        if self.down_size == 1:
            # sampling
            choice = np.random.choice(['corrupt', 'none'], 1)[0]
            if choice == 'corrupt':
                down_size = random.sample([14, 28, 56], k=1)[0]
            else:
                down_size = 112
                
        elif self.down_size == 0:
            down_size = 112
        else:
            down_size = self.down_size
        
        
        img_path = self.image_list[index]
        ind = img_path.find('faces_webface_112x112')
        img_path = img_path[ind+28:]

        label = self.label_list[index]
        HR_img, LR_img = self.loader(os.path.join(self.root, img_path), down_size)

        # random flip with ratio of 0.5
        if self.flip:
            flip = np.random.choice(2) * 2 - 1
            if flip == 1:
                HR_img = cv2.flip(HR_img, 1)
                LR_img = cv2.flip(LR_img, 1)

        if self.transform is not None:
            HR_img = self.transform(HR_img)
            LR_img = self.transform(LR_img)
            
        else:
            HR_img = torch.from_numpy(HR_img)
            LR_img = torch.from_numpy(LR_img)

        return HR_img, LR_img, label

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    root = 'D:/data/webface_align_112'
    file_list = 'D:/data/webface_align_train.list'

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    dataset = CASIAWebFace(root, file_list, transform=transform)
    trainloader = data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, drop_last=False)
    print(len(dataset))
    for data in trainloader:
        print(data[0].shape)