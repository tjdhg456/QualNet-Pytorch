import subprocess
import numpy as np
from multiprocessing import Process
import os
import argparse
import sys

sys.path.append('../')

os.environ['MKL_THREADING_LAYER'] = 'GNU'

# Basic Python Environment
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--server', type=str, default='nipa')
parser.add_argument('--exp', type=int, default=0)
args = parser.parse_args()

# OPTION - NOW
if args.exp == 0:
    '''
    toast server
    naive augmentation w/ attention distillation (cosine)
    (RUN)
    '''
    data_dir = '/data/sung/dataset/Face'
    save_dir_init = '/data/sung/checkpoint/robustness/face_recognition/A-SKD'

    # Combination
    comb_list = []
    ix = 0
    num_per_gpu = 1
    
    gpus = ['0', '1', '2', '3', '4']

    network_type = 'iresnet50'
    # condition = [[1, 'cbam'], [14, 'cbam'], [28, 'cbam'], [56, 'cbam']]
    condition = [[1, 'cbam']]

    distill_type = 'attn'
    distill_loc = 'all'
    
    for cond in condition:
        for distill_func in ['cosine']:
            for pretrained in ['true', 'false']:
                for param in [10, 30, 50, 70, 90]:
                    comb_list.append([network_type, cond[0], cond[1], param, distill_func, pretrained, '%s_%s/%s_%s_%d_%s_%.2f' %(distill_func, distill_loc, network_type, cond[1], cond[0], pretrained, param)])



elif args.exp == 1:
    '''
    toast server
    naive augmentation w/ feature distillation (cosine)
    (RUN)
    '''
    data_dir = '/data/sung/dataset/Face'
    save_dir_init = '/data/sung/checkpoint/robustness/face_recognition/F-SKD'

    # Combination
    comb_list = []
    ix = 0
    num_per_gpu = 1
    
    gpus = ['6', '7']

    network_type = 'iresnet50'
    # condition = [[1, 'cbam'], [14, 'cbam'], [28, 'cbam'], [56, 'cbam']]
    condition = [[1, 'cbam']]

    distill_type = 'feat'
    distill_loc = '3,7,21,24'
    for cond in condition:
        for distill_func in ['l1', 'l2', 'cosine']:
            for param in [50, 70, 90, 110, 130, 150]:
                comb_list.append([network_type, cond[0], cond[1], param, distill_func, '%s_%s/%s_%s_%d_%.2f' %(distill_func, distill_loc, network_type, cond[1], cond[0], param)])



else:
    raise ('Select Proper Exp Num')

comb_list = comb_list * num_per_gpu
comb_list = [comb + [index] for index, comb in enumerate(comb_list)]

arr = np.array_split(comb_list, len(gpus))
arr_dict = {}
for ix in range(len(gpus)):
    arr_dict[ix] = arr[ix]


def tr_gpu(comb, ix):
    comb = comb[ix]
    for i, comb_ix in enumerate(comb):
        gpu = gpus[ix]
        save_dir = os.path.join(save_dir_init, comb_ix[-2])
        script = 'python ../distill.py --backbone %s --down_size %d --mode %s --task_id %s \
                  --distill_type %s --cbam_distill_type both --distill_loc %s \
                  --distill_param %.2f  --distill_func %s --pretrained_student %s \
                  --save_dir %s --gpus %s --data_dir %s' %(comb_ix[0], int(comb_ix[1]), comb_ix[2], comb_ix[-2], \
                                             distill_type, distill_loc, \
                                             float(comb_ix[3]), comb_ix[4], str(comb_ix[5]), \
                                             save_dir, gpu, data_dir)
        
        subprocess.call(script, shell=True)



for ix in range(len(gpus)):
    exec('thread%d = Process(target=tr_gpu, args=(arr_dict, %d))' % (ix, ix))

for ix in range(len(gpus)):
    exec('thread%d.start()' % ix)

