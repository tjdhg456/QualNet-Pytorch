################################# Teacher ######################################## (BENGIO)
# IR - High
backbone=resnet18
python train_stage1.py --seed 0 --batch_size 64 --lr 0.01 --gpus 2 --data_type svhn --mode ir --backbone $backbone --save_dir /home/sung/checkpoint/CVPR_SVHN/qualnet_teacher/high/$backbone-ir --project_folder CVPR-ImageNet --log True


# Stage2
backbone=resnet18
teacher=/SSDb/sung/checkpoint/CVPR_SVHN/qualnet_teacher/high/$backbone-ir/last_model.pt
resol=8
python train_stage2.py --seed 0 --batch_size 64 --lr 0.01 --gpus 2 --down_size $resol --data_type svhn --mode ir --backbone $backbone --save_dir /SSDb/sung/checkpoint/CVPR_SVHN/qualnet_student/down_$resol/$backbone-ir --teacher_path $teacher --project_folder CVPR-ImageNet --log True