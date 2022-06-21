import subprocess
import numpy as np
from multiprocessing import Process
import os
import torch
import argparse
import sys
import time

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
    nipa server
    specialized network & naive augmentation
    (RUN)
    '''

    data_dir = '/home/sung/dataset/Face'
    save_dir_init = '/home/sung/checkpoint/robustness/face_recognition/base'

    # Combination
    comb_list = []
    ix = 0
    num_per_gpu = 1
    
    gpus = ['1']
    for cond in condition:
        comb_list.append([network_type, cond[0], cond[1], '%s_%s_%d' %(network_type, cond[1], cond[0])])


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
        save_dir = os.path.join(save_dir_init, comb_ix[3])
        script = 'python ../train.py --backbone %s --down_size %d --mode %s \
                  --save_dir %s --gpus %s' %(comb_ix[0], int(comb_ix[1]), comb_ix[2], \
                                             save_dir, gpu)
        
        subprocess.call(script, shell=True)


for ix in range(len(gpus)):
    exec('thread%d = Process(target=tr_gpu, args=(arr_dict, %d))' % (ix, ix))

for ix in range(len(gpus)):
    exec('thread%d.start()' % ix)

