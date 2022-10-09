################################# Teacher ######################################## (BENGIO)
'''
Train Teacher Network
'''
# IR - High
backbone=resnet50
python train_stage1.py --seed 0 --batch_size 128 --lr 0.1 --gpus 0,2 --data_dir /home/sung/dataset --ddp False --data_type imagenet --mode ir --backbone $backbone --save_dir /home/sung/checkpoint/CVPR_IMAGENET/qualnet_teacher/high/$backbone-ir --project_folder CVPR-ImageNet --log True --mixed_precision True


# Stage2
backbone=resnet50
teacher=/home/sung/checkpoint/CVPR_IMAGENET/qualnet_teacher/high/$backbone-ir/last_model.pt
resol=1
python train_stage2.py --seed 0 --batch_size 256 --gpus 0,2 --lr 0.2 --ddp True --down_size $resol --ddp False --total_epoch 1 --data_type imagenet --mode ir --backbone $backbone --save_dir /home/sung/checkpoint/CVPR_IMAGENET/qualnet_student/$resol/$backbone-ir --teacher_path $teacher --project_folder CVPR-ImageNet --log True --mixed_precision True
