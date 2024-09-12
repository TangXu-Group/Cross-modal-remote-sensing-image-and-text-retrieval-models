DATA_NAME='RSITMD'
DATA_DIR='/data1/amax/datasets/RSITMD'
IMG_DIR='/data1/amax/datasets/RSITMD/root/images'
TRAIN_FILE1='json_root/rsitmd_train.json'
VAL_FILE='json_root/rsitmd_test.json'
TEST_FILE='json_root/rsitmd_test.json'
#PRETRAIN="/data1/amax/hdb/output-5.4.1/RSITMD/CRSRT_from_scratch_plus-w-preprocess-avg_fusion_blr_0.00004000_multiscale_True_num_low_level_use_4_use_mm_fusion_True_use_alter_mlm_True_use_mim_True_use_mm_fusion_query_False/checkpoint_epoch_49_rmean_37.817109144542776.pth"
#python run_lpe.py --experiment CRSRT_LPE-plus-plus-plus-with-itmhead-wo-preprocess-avg_fusion  \
#--pretrain ${PRETRAIN} \
#--device 'cuda:0' \
#--data_name ${DATA_NAME} \
#--data_dir ${DATA_DIR} \
#--image_dir ${IMG_DIR} \
#--train_file ${TRAIN_FILE1} \
#--val_file ${VAL_FILE} \
#--test_file ${TEST_FILE} \
#--blr 4e-5 \
#--batch_size_train 64 \
#--save_epoch 10 \
#--warmup_epochs 5 \
#--epoch 200 \
#--use_mm_fusion True \
#--use_multiscale_visual_fusion True \
#--num_low_level_use 4 \
#--use_alter_mlm True \
#--use_mlm True \
#--use_mim True \
#--use_mm_fusion_query False \
#--is_use_experience True \
#--start_exp_epoch 0 \
#--cfd_epoch 0 \
#--dist_threshold 0.08 \
#--sim_threshold 0.866 \
#--beta 1 \
#--beta_rate 0.05 \
#--evaluate False

PRETRAIN="/data1/amax/hdb/Retrieval/output-5.4.1/RSITMD/CRSRT_from_scratch_plus-w-preprocess_blr_0.00004000_multiscale_True_num_low_level_use_3_use_mm_fusion_True_use_alter_mlm_True_use_mim_True_use_mm_fusion_query_True/checkpoint_epoch_68_rmean_40.11799410029498.pth"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 run_lpe.py \
--distributed True \
--experiment CRSRT_LPE-plus-plus-plus-with-itmhead-tmp-no-train-rate \
--pretrain ${PRETRAIN} \
--data_name ${DATA_NAME} \
--data_dir ${DATA_DIR} \
--image_dir ${IMG_DIR} \
--train_file ${TRAIN_FILE1} \
--val_file ${VAL_FILE} \
--test_file ${TEST_FILE} \
--blr 4e-5 \
--batch_size_train 64 \
--save_epoch 10 \
--warmup_epochs 5 \
--epoch 135 \
--use_mm_fusion True \
--use_multiscale_visual_fusion True \
--num_low_level_use 3 \
--use_alter_mlm True \
--use_mlm True \
--use_mim True \
--use_mm_fusion_query True \
--is_use_experience True \
--start_exp_epoch 0 \
--cfd_epoch 0 \
--dist_threshold 0.08 \
--sim_threshold 0.866 \
--beta 1 \
--beta_rate 0.05 \
--evaluate False

