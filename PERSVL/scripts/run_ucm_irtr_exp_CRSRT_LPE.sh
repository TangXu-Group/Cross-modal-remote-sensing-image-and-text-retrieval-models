DATA_NAME='UCM'
DATA_DIR='/data1/amax/datasets/UCM_captions'
IMG_DIR='/data1/amax/datasets/UCM_captions/root/imgs'
TRAIN_FILE1='json_root/ucm_train.json'
VAL_FILE='json_root/ucm_test.json'
TEST_FILE='json_root/ucm_test.json'
python run_lpe.py --experiment CRSRT_LPE-plus \
--distributed False \
--data_name ${DATA_NAME} \
--data_dir ${DATA_DIR} \
--image_dir ${IMG_DIR} \
--train_file ${TRAIN_FILE1} \
--val_file ${VAL_FILE} \
--test_file ${TEST_FILE} \
--blr 4e-5 \
--batch_size_train 64 \
--save_epoch 20 \
--warmup_epochs 5 \
--epoch 200 \
--use_visual_cls False \
--use_mm_fusion True \
--use_alter_visual_fusion True \
--use_multiscale_visual_fusion True \
--use_alter_mlm True \
--use_mlm True \
--use_mim True \
--low_level_stage 2 \
--is_use_experience True \
--pre_filling False \
--start_exp_epoch 50 \
--cfd_epoch 65 \
--beta 0.4 \
--evaluate False
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 run_lpe.py \
#--distributed True \
#--experiment CRSRT_LPE-plus \
#--data_name ${DATA_NAME} \
#--data_dir ${DATA_DIR} \
#--image_dir ${IMG_DIR} \
#--train_file ${TRAIN_FILE1} \
#--val_file ${VAL_FILE} \
#--test_file ${TEST_FILE} \
#--blr 4e-5 \
#--batch_size_train 32 \
#--save_epoch 10 \
#--warmup_epochs 5 \
#--epoch 200 \
#--use_visual_cls False \
#--use_mm_fusion True \
#--use_alter_visual_fusion True \
#--use_multiscale_visual_fusion True \
#--use_alter_mlm True \
#--use_mlm True \
#--use_mim True \
#--low_level_stage 2 \
#--is_use_experience True \
#--pre_filling False \
#--start_exp_epoch 50 \
#--cfd_epoch 65 \
#--beta 0.4 \
#--evaluate False