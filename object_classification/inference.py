import json
import numpy as np
import pickle
import argparse
import os
import neptune

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

from module.load_module import load_model, load_loss, load_optimizer, load_scheduler
from utility.utils import config, train_module
from utility.earlystop import EarlyStopping
import cv2
from dataset.dataset import load_data

from copy import deepcopy
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from utility.distributed import apply_gradient_allreduce, reduce_tensor
from tqdm import tqdm
from utility.noise import get_gaussian_kernel
from utility.hook import attention_manager
import torch.nn.functional as F


def overlay_heatmap(img, heatmap):
    heatmap = cv2.applyColorMap(np.uint8(heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = cv2.resize(heatmap, (img.shape[0], img.shape[0]))
    cam = heatmap + (img / 255)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    return cam


def convert(img, norm=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if norm:
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    else:
        mean, std = np.array(mean), np.array(std)
        mean, std = mean.reshape([3, 1, 1]), std.reshape([3, 1, 1])
        img = (img * std) + mean
        
    img = (np.transpose(img, (1,2,0)) * 255).astype(int)
    return img


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, option, resume, save_folder):
    # Basic Options
    resume_path = os.path.join(save_folder, 'last_dict.pt')

    num_gpu = len(option.result['train']['gpu'].split(','))

    multi_gpu = len(option.result['train']['gpu'].split(',')) > 1
    if multi_gpu:
        ddp = option.result['train']['ddp']
    else:
        ddp = False

    batch_size = option.result['train']['batch_size']

    # Load Model
    model_list, _ = load_model(option)
    criterion_list = load_loss(option)
    save_module = train_module(200, criterion_list, multi_gpu)
    save_module.import_module(resume_path)
        
    # Load Model
    for ix, model in enumerate(model_list):
        model.load_state_dict(save_module.save_dict['model'][ix])


    # Multi-Processing GPUs
    if ddp:
        setup(rank, num_gpu)
        torch.manual_seed(0)
        torch.cuda.set_device(rank)

        for ix in range(len(model_list)):
            model_list[ix].to(rank)
            model_list[ix] = DDP(model_list[ix], device_ids=[rank])
            model_list[ix] = apply_gradient_allreduce(model_list[ix])

        for ix in range(len(criterion_list)):
            criterion_list[ix].to(rank)

    else:
        for ix in range(len(model_list)):
            if multi_gpu:
                model_list[ix] = nn.DataParallel(model_list[ix]).to(rank)
            else:
                model_list[ix] = model_list[ix].to(rank)

    # Dataset and DataLoader
    val_dataset = load_data(option, data_type='val')

    if ddp:
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset=val_dataset,
                                                                     num_replicas=num_gpu, rank=rank)

        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=4*num_gpu, pin_memory=True,
                                                  sampler=val_sampler)
    else:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4*num_gpu)


    # Training
    model = model_list[0]
    hook = attention_manager(model, multi_gpu)
    
    model.eval()
    with torch.no_grad():
        for iter, val_data in enumerate(tqdm(val_loader)):            
            if iter % 10 != 0:
                continue
                
            hook.attention = []
                
            # Input Data
            HR_input, label = val_data
            
            # DownScale
            img_size= HR_input.size(-1)
            if option.result['train']['down_scale'] != 1:
                LR_input = F.interpolate(HR_input, size=int(img_size/option.result['train']['down_scale']))
                LR_input = F.interpolate(LR_input, size=img_size)
                # LR_input = get_gaussian_kernel()(LR_input)
            else:
                LR_input = HR_input
                    
            HR_input = None
            LR_input, label = LR_input.to(rank), label.to(rank)
            
            with torch.no_grad():
                _ = model(LR_input)
            
            for ix in range(LR_input.size(0)):
                image = LR_input[ix].cpu().detach().numpy()
                attention = F.sigmoid(hook.attention[0][1][1][ix]).cpu().detach().numpy()
                attention = attention[:, 4:52, 4:52]
                
                # Visualize the Image Sample
                image_ix = convert(image, norm=False)
                cv2.imwrite(os.path.join(args.visualization_dir, 'input', '%d_%04d.png' %(iter, ix)), image_ix)

                # Visualize the spatial attention
                attention = convert(attention, norm=True)
                cv2.imwrite(os.path.join(args.visualization_dir, 'attention', '%d_%04d.png' %(iter, ix)), overlay_heatmap(image_ix, attention))
                
                if ix == 10:
                    break
            
            if iter == 100:
                break    
                
    if ddp:
        cleanup()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='/home/sung/checkpoint/robustness/ECCV_imagenet/base/scale1/resnet18-CBAM')
    parser.add_argument('--down_scale', type=int, default=1)
    parser.add_argument('--visualization_dir', type=str, default='./result_vis')
    parser.add_argument('--data_dir', type=str, default='/home/sung/dataset')
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--data_type', type=str, default='imagenet')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--ddp', type=lambda x: x.lower()=='true', default=False)
    
    args = parser.parse_args()

    
    os.makedirs(os.path.join(args.visualization_dir, 'input'), exist_ok=True)
    os.makedirs(os.path.join(args.visualization_dir, 'attention'), exist_ok=True)
    
    # Configure
    resume = True

    save_folder = args.save_dir
    os.makedirs(save_folder, exist_ok=True)
    option = config(save_folder)


    # Resume Configuration
    config_path = os.path.join(save_folder, 'last_config.json')
    option.import_config(config_path)

    option.result['train']['gpu'] = args.gpu
    option.result['train']['ddp'] = args.ddp
    option.result['train']['batch_size'] = args.batch
    option.result['train']['pretrained_student'] = False
    option.result['train']['pretrained_imagenet'] = False
    option.result['train']['train_method'] = 'naive'
    option.result['data']['data_dir'] = args.data_dir


    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = option.result['train']['gpu']
    num_gpu = len(option.result['train']['gpu'].split(','))
    multi_gpu = num_gpu > 1
    if multi_gpu:
        ddp = option.result['train']['ddp']
    else:
        ddp = False

    if ddp:
        mp.spawn(main, args=(option,resume,save_folder,), nprocs=num_gpu, join=True)
    else:
        main('cuda', option, resume, save_folder)

