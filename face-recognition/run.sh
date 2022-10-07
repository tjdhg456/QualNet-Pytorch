# Teacher Training
python train_stage1.py --gpus 2 --down_size 0 --seed 0 --mode ir --backbone iresnet50 --margin_type CosFace --save_dir /data/sung/checkpoint/CVPR_FACE/qualnet_teacher/high/iresnet50-ir


# Student Training
for resol in 1 14 28 56
do
    python train_stage2.py --gpus 0 --down_size $resol --seed 0 --mode ir --backbone iresnet50 --margin_type CosFace --save_dir /data/sung/checkpoint/CVPR_FACE/qualnet_student/down_$resol/iresnet50-ir --pretrained_student True --teacher_path /data/sung/checkpoint/CVPR_FACE/qualnet_teacher/high/iresnet50-ir/last_net.ckpt
done