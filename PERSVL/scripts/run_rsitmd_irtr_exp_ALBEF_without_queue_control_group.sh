PRETRAIN_scratch='/data1/amax/pretrain_model/ALBEF-checkpoint/ALBEF_4M.pth'
DATA_NAME='RSITMD'
DATA_DIR='/data1/amax/datasets/RSITMD'
IMG_DIR='/data1/amax/datasets/RSITMD/root/images'
TRAIN_FILE1='json_root/rsitmd_train.json'
VAL_FILE='json_root/rsitmd_test.json'
TEST_FILE='json_root/rsitmd_test.json'
python run_backbone.py --experiment ALBEF_without_queue \
--pretrain_model ${PRETRAIN_scratch} \
--data_name ${DATA_NAME} \
--data_dir ${DATA_DIR} \
--image_dir ${IMG_DIR} \
--train_file ${TRAIN_FILE1} \
--val_file ${VAL_FILE} \
--test_file ${TEST_FILE} \
--use_experience False \
--pre_filling False \
--blr 8e-5 \
--batch_size_train 32 \
--save_epoch 11 \
--exp_queue_capacity 65536 \
--warmup_epochs 0 \
--epoch 8 \
--evaluate False
