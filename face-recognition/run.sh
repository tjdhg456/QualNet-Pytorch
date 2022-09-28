# Teacher Training
python train_stage1.py --gpus 3 --down_size 0 --seed 1 --mode ir --backbone iresnet50 --margin_type CosFace --save_dir /data/sung/checkpoint/lr_face_recognition/qualnet_cosface/stage1/high # High 


# Student Training
for resol in 14 28 56
do
    python train_stage2.py --gpus 3 --down_size $resol --seed 1 --mode ir --backbone iresnet50 --margin_type CosFace --save_dir /data/sung/checkpoint/lr_face_recognition/qualnet_cosface/stage2/down_$resol --pretrained_student True --teacher_path /data/sung/checkpoint/lr_face_recognition/qualnet_cosface/stage1/high/last_net.ckpt
done