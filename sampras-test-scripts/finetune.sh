# Set the path to save checkpoints
OUTPUT_DIR='/Users/apple/personal/CMU-research/VideoMAE/k400_videomae_finetune_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e100'
# path to Kinetics set (train.csv/val.csv/test.csv)
DATA_PATH='/Users/apple/personal/CMU-research/VideoMAE/shrec21_video_dataset/'
# path to pretrain model
MODEL_PATH='/Users/apple/personal/CMU-research/VideoMAE/VideoMAE_Checkpoints/ViT-B/Kinetics-400/checkpoint.pth'

# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)
OMP_NUM_THREADS=1 python ../run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 400 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 2 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt adamw \
    --lr 1e-3 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 10 \
    --dist_eval \
    --test_num_segment 5 \
    --test_num_crop 3 \
    #--enable_deepspeed 
    # --distributed False \
