#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: train.py.py
@time: 2018/12/21 17:37
@desc: train script for deep face recognition
'''

import os
import pathlib
base_folder = str(pathlib.Path(__file__).parent.resolve())
os.chdir(base_folder)
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
from backbone.irevnet import iRevNet
from margin.ArcMarginProduct import ArcMarginProduct
from margin.MultiMarginProduct import MultiMarginProduct
from margin.CosineMarginProduct import CosineMarginProduct
from margin.InnerProduct import InnerProduct
# from utility.visualize import Visualizer
from utility.log import init_log
from dataset.casia_webface import CASIAWebFace
from dataset.agedb import AgeDB30
from dataset.cfp import CFP_FP
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import time
from evaluation.eval_lfw import evaluation_10_fold, getFeatureFromTorch
import numpy as np
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
import torch.nn.functional as F

def train(args):
    # gpu init
    multi_gpus = False
    if len(args.gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # log init
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    logging = init_log(save_dir)
    _print = logging.info

    # dataset loader
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    # validation dataset
    trainset = CASIAWebFace(args.train_root, args.train_file_list, args.down_size, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=4, drop_last=False)
    

    # define backbone and margin layer
    net = iRevNet(nBlocks=[6, 16, 72, 6], nStrides=[2, 2, 2, 2],
                  nChannels=[24, 96, 384, 1536], nClasses=1000, init_ds=2,
                  dropout_rate=0., affineBN=True, in_shape=[3, 224, 224],
                  mult=4)

    if args.margin_type == 'ArcFace':
        margin = ArcMarginProduct(args.feature_dim, trainset.class_nums, s=args.scale_size)
    elif args.margin_type == 'MultiMargin':
        margin = MultiMarginProduct(args.feature_dim, trainset.class_nums, s=args.scale_size)
    elif args.margin_type == 'CosFace':
        margin = CosineMarginProduct(args.feature_dim, trainset.class_nums, s=args.scale_size, m=0.25)
    elif args.margin_type == 'Softmax':
        margin = InnerProduct(args.feature_dim, trainset.class_nums)
    elif args.margin_type == 'SphereFace':
        pass
    else:
        print(args.margin_type, 'is not available!')

    
    # define optimizers for different layer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer_ft = optim.SGD([
        {'params': net.parameters(), 'weight_decay': 5e-4},
        {'params': margin.parameters(), 'weight_decay': 5e-4}
    ], lr=0.1, momentum=0.9, nesterov=True)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[18000, 28000, 36000, 44000], \
                                                gamma=0.1)

    if multi_gpus:
        net = DataParallel(net).to(device)
        margin = DataParallel(margin).to(device)
    else:
        net = net.to(device)
        margin = margin.to(device)

    total_iters = 0

    # Run
    GOING = True
    while GOING:
        # train model
        net.train()

        since = time.time()
        for data in tqdm(trainloader):            
            run['result/iters'].log(total_iters)
            HR_img, label = data[1].to(device), data[2].to(device)                  
                        
            raw_logits, out_bij = net(HR_img, train=True)
            out = margin(raw_logits, label)
            
            # Loss
            cri_loss = criterion(out, label)
            total_loss = cri_loss


            # Optim
            optimizer_ft.zero_grad()
            total_loss.backward()
            optimizer_ft.step()


            # print train information
            if total_iters % 100 == 0:
                # current training accuracy
                _, predict = torch.max(out.data, 1)
                total = label.size(0)

                correct = (np.array(predict.cpu()) == np.array(label.data.cpu())).sum()
                time_cur = (time.time() - since) / 100
                since = time.time()
                run['result/train_loss'].log(total_loss.item())
                run['result/train_accuracy'].log(correct/total)
                _print("Iters: {:0>6d}, loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}".format(total_iters, total_loss.item(), correct/total, time_cur, exp_lr_scheduler.get_lr()[0]))


            # save model
            if total_iters % args.save_freq == 0:
                msg = 'Saving checkpoint: {}'.format(total_iters)
                _print(msg)
                if multi_gpus:
                    net_state_dict = net.module.state_dict()
                    margin_state_dict = margin.module.state_dict()
                else:
                    net_state_dict = net.state_dict()
                    margin_state_dict = margin.state_dict()
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                    
                torch.save({
                    'iters': total_iters,
                    'net_state_dict': net_state_dict},
                    os.path.join(save_dir, 'Iter_%06d_net.ckpt' % total_iters))
                torch.save({
                    'iters': total_iters,
                    'net_state_dict': margin_state_dict},
                    os.path.join(save_dir, 'Iter_%06d_margin.ckpt' % total_iters))
            
            
            run['debug/lr'].log(exp_lr_scheduler.get_lr()[0])
            exp_lr_scheduler.step()
            
            # Stop
            if total_iters == 47000:
                GOING = False
                break
        
            # Next Iterations
            total_iters += 1
            
            

    # Save Last Epoch
    msg = 'Saving checkpoint: {}'.format(total_iters)
    _print(msg)
    if multi_gpus:
        net_state_dict = net.module.state_dict()
        margin_state_dict = margin.module.state_dict()
    else:
        net_state_dict = net.state_dict()
        margin_state_dict = margin.state_dict()
        
    torch.save({
        'iters': total_iters,
        'net_state_dict': net_state_dict},
        os.path.join(save_dir, 'last_net.ckpt'))
    torch.save({
        'iters': total_iters,
        'net_state_dict': margin_state_dict},
        os.path.join(save_dir, 'last_margin.ckpt'))


    # test dataset
    net.eval()
    
    print('Evaluation on LFW, AgeDB-30. CFP')
    os.makedirs(os.path.join(args.save_dir, 'result'), exist_ok=True)
    for down_size in [14, 28, 56, 112]:
        # lfwdataset = LFW(args.lfw_test_root, args.lfw_file_list, down_size, transform=transform)
        agedbdataset = AgeDB30(args.agedb_test_root, args.agedb_file_list, down_size, transform=transform)
        cfpfpdataset = CFP_FP(args.cfpfp_test_root, args.cfpfp_file_list, down_size, transform=transform)
        
        # lfwloader = torch.utils.data.DataLoader(lfwdataset, batch_size=128, shuffle=False, num_workers=4, drop_last=False)
        agedbloader = torch.utils.data.DataLoader(agedbdataset, batch_size=128, shuffle=False, num_workers=4, drop_last=False)
        cfpfploader = torch.utils.data.DataLoader(cfpfpdataset, batch_size=128, shuffle=False, num_workers=4, drop_last=False)

        # # test model on lfw
        # getFeatureFromTorch(os.path.join(args.save_dir, 'result/cur_lfw_result.mat'), net, device, lfwdataset, lfwloader)
        # lfw_accs = evaluation_10_fold(os.path.join(args.save_dir, 'result/cur_lfw_result.mat'))
        # run['result/lfw_acc_%d' %down_size].log(np.mean(lfw_accs) * 100)
        
        # test model on AgeDB30
        getFeatureFromTorch(os.path.join(args.save_dir, 'result/cur_agedb30_result.mat'), net, device, agedbdataset, agedbloader)
        age_accs = evaluation_10_fold(os.path.join(args.save_dir, 'result/cur_agedb30_result.mat'))
        run['result/age_acc_%d' %down_size].log(np.mean(age_accs) * 100)

        # test model on CFP-FP
        getFeatureFromTorch(os.path.join(args.save_dir, 'result/cur_cfpfp_result.mat'), net, device, cfpfpdataset, cfpfploader)
        cfp_accs = evaluation_10_fold(os.path.join(args.save_dir, 'result/cur_cfpfp_result.mat'))
        run['result/cfp_acc_%d' %down_size].log(np.mean(cfp_accs) * 100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    parser.add_argument('--data_dir', type=str, default='/data/sung/dataset/Face')
    parser.add_argument('--save_dir', type=str, default='/data/sung/checkpoint/robustness/face_recognition/qualnet_stage1', help='model save dir')
    parser.add_argument('--down_size', type=int, default=0) # 1 : all type, 0 : high, others : low
    parser.add_argument('--margin_type', type=str, default='CosFace', help='ArcFace, CosFace, SphereFace, MultiMargin, Softmax')
    parser.add_argument('--feature_dim', type=int, default=512, help='feature dimension, 128 or 512')
    parser.add_argument('--scale_size', type=float, default=30.0, help='scale size')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--save_freq', type=int, default=10000, help='save frequency')
    parser.add_argument('--gpus', type=str, default='0', help='model prefix')
    args = parser.parse_args()

    # Path
    args.train_root = os.path.join(args.data_dir, 'faces_webface_112x112/image')
    args.train_file_list = os.path.join(args.data_dir, 'faces_webface_112x112/train.list')
    args.lfw_test_root = os.path.join(args.data_dir, 'evaluation/lfw')
    args.lfw_file_list = os.path.join(args.data_dir, 'evaluation/lfw.txt')
    args.agedb_test_root = os.path.join(args.data_dir, 'evaluation/agedb_30')
    args.agedb_file_list = os.path.join(args.data_dir, 'evaluation/agedb_30.txt')
    args.cfpfp_test_root = os.path.join(args.data_dir, 'evaluation/cfp_fp')
    args.cfpfp_file_list = os.path.join(args.data_dir, 'evaluation/cfp_fp.txt')

    # Logger
    import neptune.new as neptune
    monitor_hardware = True
    mode = 'async'
    token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5MTQ3MjY2Yy03YmM4LTRkOGYtOWYxYy0zOTk3MWI0ZDY3M2MifQ=='
    run = neptune.init('sunghoshin/ROBUSTNESS', api_token=token,
                capture_stdout=monitor_hardware,
                capture_stderr=monitor_hardware,
                capture_hardware_metrics=monitor_hardware,
                mode=mode
                )

    for key, value in vars(args).items():
        run[key] = value
    
    run['task'] = 'face_recognition_qualnet_stage1'
    train(args)