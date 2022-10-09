from torchvision.transforms import transforms
import torchvision
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from random import shuffle
import torch
import random
from glob import glob
from PIL import Image
import pickle
import cv2


def load_imagenet(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    scale = 256
    tr_resize, val_resize = transforms.RandomSizedCrop(scale), transforms.Compose([transforms.Resize((int(scale * 256 / 224), int(scale * 256 / 224))), transforms.CenterCrop(scale)])
    
    tr_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
        
      
    tr_dataset = Base_Dset(os.path.join(args.data_dir, 'imagenet', 'train'), resize=tr_resize, transform=tr_transform, image_list=None, resolution=args.down_size)
    val_dataset_256 = Base_Dset(os.path.join(args.data_dir, 'imagenet', 'val'), resize=val_resize, transform=val_transform, image_list=None, resolution=256)
    val_dataset_128 = Base_Dset(os.path.join(args.data_dir, 'imagenet', 'val'), resize=val_resize, transform=val_transform, image_list=None, resolution=128)
    val_dataset_64 = Base_Dset(os.path.join(args.data_dir, 'imagenet', 'val'), resize=val_resize, transform=val_transform, image_list=None, resolution=64)
    val_dataset_32 = Base_Dset(os.path.join(args.data_dir, 'imagenet', 'val'), resize=val_resize, transform=val_transform, image_list=None, resolution=32)
    val_dataset_16 = Base_Dset(os.path.join(args.data_dir, 'imagenet', 'val'), resize=val_resize, transform=val_transform, image_list=None, resolution=16)
    return tr_dataset, [val_dataset_256, val_dataset_128, val_dataset_64, val_dataset_32, val_dataset_16]



def load_data(args, data_type='train'):
    if args.data_type == 'imagenet':
        tr_d, val_d = load_imagenet(args)
    else:
        raise('select appropriate dataset')

    if data_type == 'train':
        return tr_d
    else:
        return val_d


class Base_Dset(Dataset):
    def __init__(self, data_dir, resize=None, transform=None, image_list=None, resolution=0):
        # Generate Class Lists
        if image_list is not None:
            class_list = np.unique([image_path.split('/')[-2] for image_path in image_list])
            class_list = sorted(class_list)
            self.label_dict = {class_name: class_index for class_index, class_name in enumerate(class_list)}
        else:
            class_list = sorted(os.listdir(data_dir))
            self.label_dict = {class_name: class_index for class_index, class_name in enumerate(class_list)}
        
        
        # Select Image Lists
        if image_list is not None:
            self.image_list = image_list
        else:
            self.image_list = []
            for class_name in class_list:
                self.image_list += glob(os.path.join(data_dir, class_name, '*.JPEG'))
                    
                
        # Transform and Augmentation
        self.resize = resize
        self.transform = transform
        self.resolution = resolution
        
        
    def __len__(self):
        return len(self.image_list)
    
    
    def __getitem__(self, index):
        # Resolution
        if self.resolution == 1:
            down_size = random.sample([256, 128, 64, 32], k=1)[0]
        elif self.resolution == 0:
            down_size = 256
        else:
            down_size = self.resolution
        
        
        # Load Image and Label
        image_path = self.image_list[index]
        label_name = image_path.split('/')[-2]
        label = self.label_dict[label_name]
        
        
        # Augment HR / LR images        
        HR_img, LR_img = image_loader(image_path, down_size, self.resize)

        HR_img = self.transform(HR_img)
        LR_img = self.transform(LR_img)
        return HR_img, LR_img, label
    

def image_loader(path, down_size, resize):
    try:
        with open(path, 'rb') as f:
            high_img = cv2.imread(path)
            
            # convert cv2 -> PIL -> HR to LR -> cv2
            high_img = cv2.cvtColor(high_img, cv2.COLOR_BGR2RGB)
            high_img = Image.fromarray(high_img)
            high_img = resize(high_img)
            high_img = np.asarray(high_img)
            high_img = cv2.cvtColor(high_img, cv2.COLOR_RGB2BGR)

            if len(high_img.shape) == 2:
                high_img = np.stack([high_img] * 3, 2)
            
            if down_size != 256:
                down_img = cv2.resize(high_img, dsize=(down_size, down_size), interpolation=cv2.INTER_NEAREST)
                down_img = cv2.resize(down_img, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
            else:
                down_img = high_img
                
                
            # convert cv2 -> PIL
            high_img = cv2.cvtColor(high_img, cv2.COLOR_BGR2RGB)
            high_img = Image.fromarray(high_img)
            
            down_img = cv2.cvtColor(down_img, cv2.COLOR_BGR2RGB)
            down_img = Image.fromarray(down_img)
            return high_img, down_img
        
    except IOError:
        print('Cannot load image ' + path)
        
        
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    
    # args.data_dir = '/data/sung/dataset'
    # args.down_size = 1
    # tr_d, val_d_list = load_small_imagenet(args)
    # out2 = val_d_list[0].__getitem__(1)