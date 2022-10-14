import json
import numpy as np
import pickle
import argparse
import os
import pathlib
base_folder = str(pathlib.Path(__file__).parent.resolve())
os.chdir(base_folder)

import neptune.new as neptune
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from module.load_model import load_student_model
from module.trainer import train_stage2, validation
from dataset.dataset import load_data
from copy import deepcopy
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from utility.distributed import apply_gradient_allreduce, reduce_tensor
import pathlib
import random
    

def freeze_batchnorm(model):
    for name, layer in model._modules.items():
        if isinstance(layer, nn.Sequential):
            freeze_batchnorm(layer)
        else:
            for name, layer2 in layer._modules.items():
                if isinstance(layer2, nn.Sequential):
                    freeze_batchnorm(layer2)
                else:
                    if isinstance(layer2, nn.BatchNorm2d):
                        if hasattr(layer2, 'weight'):
                            layer2.weight.requires_grad_(False)
                        if hasattr(layer2, 'bias'):
                            layer2.bias.requires_grad_(False)
                            
                            
def set_random_seed(seed_value, use_cuda=True):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    os.environ['PYTHONHASHSEED'] = str(seed_value) # Python hash buildin
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False


def setup(rank, world_size, master_port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port

    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def main(rank, args, save_folder, log, master_port):
    # Basic Options
    total_epoch = args.total_epoch

    # GPU Configuration
    num_gpu = len(args.gpus.split(','))
    multi_gpu = num_gpu > 1
    if multi_gpu:
        ddp = args.ddp
    else:
        ddp = False

    # Logger
    if (rank == 0) or (rank == 'cuda'):
        token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5MTQ3MjY2Yy03YmM4LTRkOGYtOWYxYy0zOTk3MWI0ZDY3M2MifQ=='

        if log:
            mode = 'async'
        else:
            mode = 'debug'

        monitoring_hardware = True
        run = neptune.init('sunghoshin/%s' %args.project_folder, api_token=token,
                            capture_stdout=monitoring_hardware,
                            capture_stderr=monitoring_hardware,
                            capture_hardware_metrics=monitoring_hardware,
                            mode = mode
                            )

        for key, value in vars(args).items():
            run[key] = value
        
    else:
        run = None

    # Load Model
    model, decoder = load_student_model(args)
    
    if args.pretrained_student:
        model.load_state_dict(torch.load(args.teacher_path)['model'])
        freeze_batchnorm(model)
        
    decoder.load_state_dict(torch.load(args.teacher_path)['decoder'])
    for param in decoder.parameters():
        param.requires_grad = False
    
    criterion = nn.CrossEntropyLoss()


    # Multi-Processing GPUs
    if ddp:
        setup(rank, num_gpu, master_port)
        torch.cuda.set_device(rank)

        model.to(rank)
        model = DDP(model, device_ids=[rank])
        model = apply_gradient_allreduce(model)
        
        decoder.to(rank)
        criterion.to(rank)

    else:
        if multi_gpu:
            model = nn.DataParallel(model).to(rank)
            decoder = decoder.to(rank)
        else:
            model = model.to(rank)
            decoder = decoder.to(rank)
            
            

    # Optimizer and Scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)

    # Dataset and DataLoader
    tr_dataset = load_data(args, data_type='train')
    val_dataset_list = load_data(args, data_type='val')

    # Data Loader
    if ddp:
        # tr_dataset
        tr_sampler = torch.utils.data.distributed.DistributedSampler(dataset=tr_dataset,
                                                                     num_replicas=num_gpu, rank=rank)
        tr_loader = torch.utils.data.DataLoader(dataset=tr_dataset, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=args.num_workers, pin_memory=True,
                                                  sampler=tr_sampler)
        
        # val_dataset
        val_loader_list = []        
        for val_dataset in val_dataset_list:
            val_loader_list.append( torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers, pin_memory=True,
                                    sampler=torch.utils.data.distributed.DistributedSampler(dataset=val_dataset, num_replicas=num_gpu, rank=rank)))
    
    else:
        tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
        
        val_loader_list = []
        for val_dataset in val_dataset_list:
            val_loader_list.append( DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers) )  


    # Mixed Precision
    mixed_precision = args.mixed_precision
    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None        

    # Run
    for epoch in range(total_epoch):
        scheduler.step()
        save_param = train_stage2(args, rank, epoch, model, decoder, criterion, optimizer, multi_gpu, tr_loader, scaler, run, save_folder)
               
        # Log Learning Rate
        if run is not None:
            for param_group in optimizer.param_groups:
                run['debug/current_lr'].log(param_group['lr'])
                
        # Save the last-epoch module
        if (rank == 0) or (rank == 'cuda'):
            torch.save(save_param, os.path.join(args.save_dir, 'last_model.pt'))

    
        # Validation
        if epoch % args.save_epoch == 0:
            if args.down_size == 0:
                result = validation(args, rank, epoch, model, multi_gpu, val_loader_list[0], 256, run)
            else:
                resolution = [256, 128, 64, 32]
                for ix, val_loader in enumerate(val_loader_list):
                    result = validation(args, rank, epoch, model, multi_gpu, val_loader, resolution[ix], run)

    if ddp:
        cleanup()
    return None


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='/data/sung/checkpoint/imp')
    parser.add_argument('--save_epoch', type=int, default=10)
    parser.add_argument('--total_epoch', type=int, default=90)
    
    parser.add_argument('--data_dir', type=str, default='/data/sung/dataset')
    parser.add_argument('--data_type', type=str, default='imagenet')
    
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--mode', type=str, default='ir')
    
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--down_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
        
    parser.add_argument('--log', type=lambda x: x.lower()=='true', default=False)
    parser.add_argument('--project_folder', type=str, default='ROBUSTNESS')
    
    parser.add_argument('--gpus', type=str, default='2')
    parser.add_argument('--ddp', type=lambda x: x.lower()=='true', default=False)
    parser.add_argument('--mixed_precision', type=lambda x: x.lower()=='true', default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task', type=str, default='qualnet_stage2')
    
    parser.add_argument('--teacher_path', type=str, default='//')
    args = parser.parse_args()

    args.num_classes = 1000
    args.pretrained_student = True
    
    # Configure
    save_folder = args.save_dir
    os.makedirs(save_folder, exist_ok=True)
    

    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    num_gpu = len(args.gpus.split(','))
    multi_gpu = num_gpu > 1
    if multi_gpu:
        ddp = args.ddp
    else:
        ddp = False

    master_port = str(random.randint(50, 1000))
    
    # Set Seed
    set_random_seed(args.seed)
    
    # ImageNet Train
    if ddp:
        mp.spawn(main, args=(args, save_folder, args.log, master_port, ), nprocs=num_gpu, join=True)
    else:
        main('cuda', args, save_folder, args.log, master_port)
    
    exit()