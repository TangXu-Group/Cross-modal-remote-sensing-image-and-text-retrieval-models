DATA_NAME='RSITMD'
DATA_DIR='/data1/amax/datasets/RSITMD'
IMG_DIR='/data1/amax/datasets/RSITMD/root/images'
TRAIN_FILE1='json_root/rsitmd_train.json'
VAL_FILE='json_root/rsitmd_test.json'
TEST_FILE='json_root/rsitmd_test.json'
#python run_backbone.py --experiment CRSRT_from_scratch_plus-wo-preprocess-bs64 \
#--device 'cuda:0' \
#--data_name ${DATA_NAME} \
#--data_dir ${DATA_DIR} \
#--image_dir ${IMG_DIR} \
#--train_file ${TRAIN_FILE1} \
#--val_file ${VAL_FILE} \
#--test_file ${TEST_FILE} \
#--is_use_experience False \
#--blr 4e-5 \
#--batch_size_train 64 \
#--save_epoch 10 \
#--warmup_epochs 5 \
#--epoch 135 \
#--use_mm_fusion False \
#--use_multiscale_visual_fusion True \
#--num_low_level_use 4 \
#--use_alter_mlm False \
#--use_mlm False \
#--use_mim False \
#--use_mm_fusion_query False \
#--evaluate False
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 run_backbone.py \
--experiment CRSRT_from_scratch_plus-w-preprocess-tmp \
--data_name ${DATA_NAME} \
--data_dir ${DATA_DIR} \
--image_dir ${IMG_DIR} \
--train_file ${TRAIN_FILE1} \
--val_file ${VAL_FILE} \
--test_file ${TEST_FILE} \
--is_use_experience False \
--blr 4e-5 \
--batch_size_train 64 \
--save_epoch 10 \
--warmup_epochs 5 \
--epoch 135 \
--use_mm_fusion True \
--use_multiscale_visual_fusion True \
--num_low_level_use 4 \
--use_alter_mlm True \
--use_mlm True \
--use_mim False \
--use_mm_fusion_query True \
--evaluate False

