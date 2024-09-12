DATA_NAME='UCM'
DATA_DIR='/data1/amax/datasets/UCM_captions'
IMG_DIR='/data1/amax/datasets/UCM_captions/root/imgs'
TRAIN_FILE1='json_root/ucm_train.json'
VAL_FILE='json_root/ucm_test.json'
TEST_FILE='json_root/ucm_test.json'
python run_backbone.py --experiment CRSRT_control_group \
--data_name ${DATA_NAME} \
--data_dir ${DATA_DIR} \
--image_dir ${IMG_DIR} \
--train_file ${TRAIN_FILE1} \
--val_file ${VAL_FILE} \
--test_file ${TEST_FILE} \
--use_experience False \
--pre_filling False \
--blr 8e-3 \
--batch_size_train 64 \
--save_epoch 11 \
--warmup_epochs 0 \
--epoch 100 \
--use_visual_cls True \
--low_level_stage 2 \
--MM_fusion_start_layer 6 \
--evaluate False \


