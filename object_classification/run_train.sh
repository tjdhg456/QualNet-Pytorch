################################# Teacher ######################################## (BENGIO)
'''
Train Teacher Network
'''
# IR - High
backbone=resnet50
python train_stage1.py --seed 0 --batch_size 256 --lr 0.2 --gpus 0,2 --ddp True --data_type imagenet --mode ir --backbone $backbone --save_dir /data/sung/checkpoint/CVPR_IMAGENET/qualnet_teacher/high/$backbone-ir --project_folder CVPR-ImageNet --log True --mixed_precision True


# Stage2
backbone=resnet50
teacher=/home/sung/checkpoint/CVPR_IMAGENET/qualnet_teacher/high/$backbone-ir/last_model.pt
resol=1
python train_stage2.py --seed 0 --batch_size 256 --gpus 0,2 --lr 0.2 --ddp True --down_size $resol --ddp True --data_type imagenet --mode ir --backbone $backbone --save_dir /home/sung/checkpoint/CVPR_IMAGENET/qualnet_student/down_$resol/$backbone-ir --teacher_path $teacher --project_folder CVPR-ImageNet --log True --mixed_precision True
