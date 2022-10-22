import numpy as np
import torch
from tqdm import tqdm
import os
from utility.distributed import apply_gradient_allreduce, reduce_tensor
import torch.nn as nn
from copy import deepcopy
import torch.distributed as dist
from copy import deepcopy
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def un_normalize(image, mu=torch.tensor([0.485, 0.456, 0.406]).float(), std=torch.tensor([0.229, 0.224, 0.225]).float()):
    device = image.device
    image = image.permute(0, 2, 3, 1) * std.to(device) + mu.to(device)
    image = image.permute(0, 3, 1, 2)
    return image
    
    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



def train_stage1(args, rank, epoch, model_wrapper, criterion, optimizer, multi_gpu, tr_loader, scaler, logger, save_folder):
    # GPU setup
    num_gpu = len(args.gpus.split(','))

    # For Log
    mean_loss_cls = 0.
    mean_acc1 = 0.
    mean_acc5 = 0.

    # Freeze !
    model_wrapper.train()
        
    # Run
    for iter, tr_data in enumerate(tqdm(tr_loader)):
        # Dataset
        HR_img, label = tr_data
        HR_img, label = HR_img.to(rank), label.to(rank)

        # Forward
        optimizer.zero_grad()
            
        if scaler is not None:
            with torch.cuda.amp.autocast():
                out, HR_img_gen = model_wrapper(HR_img, train=True)
                loss_cls = criterion(out, label)
                loss_recon = F.l1_loss(torch.clip(HR_img_gen, 0, 1), un_normalize(HR_img))
                loss_total = loss_cls + loss_recon * (1.0)

                scaler.scale(loss_total).backward()
                scaler.step(optimizer)
                scaler.update()
                
        else:
            out, HR_img_gen = model_wrapper(HR_img, train=True)
            loss_cls = criterion(out, label)
            loss_recon = F.l1_loss(torch.clip(HR_img_gen, 0, 1), un_normalize(HR_img))
            loss_total = loss_cls + loss_recon * (1.0)
            
            loss_total.backward()
            optimizer.step()
            
            
        # Metrics
        acc_result = accuracy(out, label, topk=(1, 5))
        
        if (num_gpu > 1) and args.ddp:
            mean_loss_cls += reduce_tensor(loss_total.data, num_gpu).item()
            mean_acc1 += reduce_tensor(acc_result[0], num_gpu)
            mean_acc5 += reduce_tensor(acc_result[1], num_gpu)

        else:
            mean_loss_cls += loss_total.item()
            mean_acc1 += acc_result[0]
            mean_acc5 += acc_result[1]
        
        del out, loss_cls, loss_total

    # Train Result
    mean_acc1 /= len(tr_loader)
    mean_acc5 /= len(tr_loader)
    mean_loss_cls /= len(tr_loader)

    # Model Params
    if multi_gpu:
        model_param = deepcopy(model_wrapper.module.model.state_dict())
        decoder_param = deepcopy(model_wrapper.module.decoder.state_dict())
    else:
        model_param = deepcopy(model_wrapper.model.state_dict())
        decoder_param = deepcopy(model_wrapper.decoder.state_dict())

    # Save
    save_module = {'model': model_param, 'decoder': decoder_param, 'optimizer': deepcopy(optimizer.state_dict()), 'save_epoch': epoch}

    if (rank == 0) or (rank == 'cuda'):
        # Logging
        print('Epoch-(%d/%d) - tr_ACC@1: %.2f, tr_ACC@5-%.2f, tr_loss_cls:%.3f' %(epoch, args.total_epoch, mean_acc1, mean_acc5, mean_loss_cls))
        logger['result/tr_loss_cls'].log(mean_loss_cls)
        logger['result/tr_acc1'].log(mean_acc1)
        logger['result/tr_acc5'].log(mean_acc5)
        logger['result/epoch'].log(epoch)
        
        # Save
        if epoch % args.save_epoch == 0:
            torch.save(save_module, os.path.join(save_folder, 'epoch%d_model.pt' %epoch))

    if multi_gpu and (args.ddp):
        dist.barrier()

    return save_module


def train_stage2(args, rank, epoch, model, decoder, criterion, optimizer, multi_gpu, tr_loader, scaler, logger, save_folder):
    # GPU setup
    num_gpu = len(args.gpus.split(','))

    # For Log
    mean_loss_cls = 0.
    mean_acc1 = 0.
    mean_acc5 = 0.

    # Freeze !
    model.train()    
    decoder.eval()
    
    # Run
    for iter, tr_data in enumerate(tqdm(tr_loader)):
        # Dataset
        HR_input, label = tr_data
        
        # DownScale
        img_size= HR_input.size(-1)
        LR_input = F.interpolate(HR_input, size=int(args.down_size))
        LR_input = F.interpolate(LR_input, size=img_size)

        HR_input, LR_input, label = HR_input.to(rank), LR_input.to(rank), label.to(rank)

        # Forward
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                # LR output
                LR_output, out_bij = model(LR_input, train=True)
                
                with torch.no_grad():
                    HR_img_gen = decoder.inverse(out_bij)
                        
                loss_cls = criterion(LR_output, label)
                loss_recon = F.l1_loss(torch.clip(HR_img_gen, 0, 1), un_normalize(HR_input))
                loss_total = loss_cls + loss_recon * 1.0
                
                # Backward
                scaler.scale(loss_total).backward()
                scaler.step(optimizer)
                scaler.update()
                
        else:
            # LR output
            LR_output, out_bij = model(LR_input, train=True)
            
            with torch.no_grad():
                HR_img_gen = decoder.inverse(out_bij)
                    
            loss_cls = criterion(LR_output, label)
            loss_recon = F.l1_loss(torch.clip(HR_img_gen, 0, 1), un_normalize(HR_input))
            loss_total = loss_cls + loss_recon * 1.0
                    
            loss_total.backward()
            optimizer.step()
            

        # Metrics
        acc_result = accuracy(LR_output, label, topk=(1, 5))
        
        if (num_gpu > 1) and args.ddp:
            mean_loss_cls += reduce_tensor(loss_total.data, num_gpu).item()
            mean_acc1 += reduce_tensor(acc_result[0], num_gpu)
            mean_acc5 += reduce_tensor(acc_result[1], num_gpu)

        else:
            mean_loss_cls += loss_total.item()
            mean_acc1 += acc_result[0]
            mean_acc5 += acc_result[1]
        

    # Train Result
    mean_acc1 /= len(tr_loader)
    mean_acc5 /= len(tr_loader)
    mean_loss_cls /= len(tr_loader)

    # Model Params
    if multi_gpu:
        model_param = deepcopy(model.module.state_dict())
    else:
        model_param = deepcopy(model.state_dict())

    # Save
    save_module = {'model': model_param, 'optimizer': deepcopy(optimizer.state_dict()), 'save_epoch': epoch}

    if (rank == 0) or (rank == 'cuda'):
        # Logging
        print('Epoch-(%d/%d) - tr_ACC@1: %.2f, tr_ACC@5-%.2f, tr_loss_cls:%.3f' %(epoch, args.total_epoch, mean_acc1, mean_acc5, mean_loss_cls))
        logger['result/tr_loss_cls'].log(mean_loss_cls)
        logger['result/tr_acc1'].log(mean_acc1)
        logger['result/tr_acc5'].log(mean_acc5)
        logger['result/epoch'].log(epoch)
        
        # Save
        if epoch % args.save_epoch == 0:
            torch.save(save_module, os.path.join(save_folder, 'epoch%d_model.pt' %epoch))

    if multi_gpu and (args.ddp):
        dist.barrier()

    return save_module


def validation(args, rank, epoch, model, multi_gpu, val_loader, logger):
    # GPU setup
    num_gpu = len(args.gpus.split(','))

    # For Log
    mean_acc1 = 0.
    mean_acc5 = 0.

    # Freeze !
    model.eval()

    # Run
    with torch.no_grad():
        for iter, val_data in enumerate(tqdm(val_loader)):
            # Dataset
            input, label = val_data
            input, label = input.to(rank), label.to(rank)
            
            # DownScale
            if args.down_size != 0:
                img_size= input.size(-1)
                input = F.interpolate(input, size=int(args.down_size))
                input = F.interpolate(input, size=img_size)
                
            output = model(input, train=False)
            
            # Metrics
            acc_result = accuracy(output, label, topk=(1, 5))
            
            if (num_gpu > 1) and args.ddp:
                mean_acc1 += reduce_tensor(acc_result[0], num_gpu)
                mean_acc5 += reduce_tensor(acc_result[1], num_gpu)

            else:
                mean_acc1 += acc_result[0]
                mean_acc5 += acc_result[1]
        

    # Val Result
    mean_acc1 /= len(val_loader)
    mean_acc5 /= len(val_loader)


    # Logging
    if (rank == 0) or (rank == 'cuda'):
        print('Epoch-(%d/%d) - val_ACC@1: %.2f, val_ACC@5-%.2f' % (epoch, args.total_epoch, \
                                                                    mean_acc1, mean_acc5))
        logger['result/val_acc1_down%d' %args.down_size].log(mean_acc1)
        logger['result/val_acc5_down%d' %args.down_size].log(mean_acc5)

    result = {'acc1':mean_acc1, 'acc5':mean_acc5}

    if multi_gpu and (args.ddp):
        dist.barrier()

    return result