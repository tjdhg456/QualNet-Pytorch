# Student Training for CBAM (A-SKD)
backbone=resnet50
teacher=/data/sung/checkpoint/CVPR_IMAGENET/hyperparam/high/resnet50-cbam/last_model.pt
resol=32
for param in 50 40 30 20 10
do
    python train_student.py --task hyperparam --seed 0 --gpus 4 --data_type small_imagenet --distill_type A_SKD --distill_param $param --teacher_backbone resnet50 --teacher_path $teacher \
                            --down_size $resol --mode cbam --backbone $backbone --save_dir /data/sung/checkpoint/CVPR_IMAGENET/hyperparam/A_SKD_$param \
                            --project_folder CVPR-ImageNet --log True --mixed_precision True
done


# Student Training for IR (F-SKD)
backbone=resnet50
teacher=/data/sung/checkpoint/CVPR_IMAGENET/hyperparam/high/resnet50-ir/last_model.pt
resol=32
for param in 10 8 6 4 2
do
    python train_student.py --task hyperparam --seed 0 --gpus 3 --data_type small_imagenet --distill_type F_SKD_BLOCK --distill_param $param --teacher_backbone resnet50 --teacher_path $teacher \
                            --down_size $resol --mode ir --backbone $backbone --save_dir /data/sung/checkpoint/CVPR_IMAGENET/hyperparam/F_SKD_$param \
                            --project_folder CVPR-ImageNet --log True --mixed_precision True
done



# Student Training for IR (RKD)
backbone=resnet50
teacher=/data/sung/checkpoint/CVPR_IMAGENET/hyperparam/high/resnet50-ir/last_model.pt
resol=32
for param in 10 8 6 4 2
do
    python train_student.py --task hyperparam --seed 0 --gpus 3 --data_type small_imagenet --distill_type RKD --distill_param $param --teacher_backbone resnet50 --teacher_path $teacher \
                            --down_size $resol --mode ir --backbone $backbone --save_dir /data/sung/checkpoint/CVPR_IMAGENET/hyperparam/RKD_$param \
                            --project_folder CVPR-ImageNet --log True --mixed_precision True
done



# Student Training for IR (F_KD)
backbone=resnet50
teacher=/data/sung/checkpoint/CVPR_IMAGENET/hyperparam/high/resnet50-ir/last_model.pt
resol=32
for param in 2 1.6 1.2 0.8 0.4
do
    python train_student.py --task hyperparam --seed 0 --gpus 3 --data_type small_imagenet --distill_type F_KD --distill_param $param --teacher_backbone resnet50 --teacher_path $teacher \
                            --down_size $resol --mode ir --backbone $backbone --save_dir /data/sung/checkpoint/CVPR_IMAGENET/hyperparam/F_KD_$param \
                            --project_folder CVPR-ImageNet --log True --mixed_precision True
done