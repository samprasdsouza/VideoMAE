# Set the path to save video
OUTPUT_DIR='TODO-2/VideoMAE/demo/vis_k400_1_0.9'
# path to video for visualization
VIDEO_PATH='TODO-2/class_mask_standardized_x_gray.mp4'
# path to pretrain model
MODEL_PATH='VideoMAE_Checkpoints/ViT-B/Kinetics-400/checkpoint-10.pth'

python3 run_videomae_vis.py \
    --mask_ratio 0.9 \
    --mask_type tube \
    --decoder_depth 4 \
    --model pretrain_videomae_base_patch16_224 \
    ${VIDEO_PATH} ${OUTPUT_DIR} ${MODEL_PATH}