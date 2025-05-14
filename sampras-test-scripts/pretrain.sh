# Set the path to save checkpoints
OUTPUT_DIR='/shared/home/v_sampras_dsouza/my_scratch/VideoMAE/VideoMAE/k400_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e100'
# Set the path to Kinetics train set. 
DATA_PATH='/shared/scratch/0/home/v_sampras_dsouza/shrec_21_video_dataset/train.csv'

# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)
python ../run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 1 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 2 \
        --save_ckpt_freq 20 \
        --epochs 11 \
        --weight_decay 0.05\
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR}