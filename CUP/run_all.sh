cd ..
CARD=0
#!/bin/bash

source  /home/amax/anaconda3/etc/profile.d/conda.sh
conda deactivate



echo ""
echo ""
echo "****************** Activating conda environment  zzzz ******************"
conda activate zzzz


echo ""
echo "****************** Testing ******************"
echo "****************** Testing ******************"
echo ""
echo ""

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSITMD task_finetune_irtr_rsitmd_randaug_ViTB32 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 prompt_length=8 weight_margin=0.015  warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1 test_only=True load_path=<ckpt_path>



batch_size=160

cd /home/amax/wyj/CUP_Test

echo ""
echo "****************** Training******************"
echo "****************** Training ******************"
echo ""
echo ""

echo ""
echo "****************** RSITMD_captions learning_rate=5e-4******************"
echo ""


CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSITMD task_finetune_irtr_rsitmd_randaug_ViTB32 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 prompt_length=1 weight_margin=0.015  warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSITMD task_finetune_irtr_rsitmd_randaug_ViTB32 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 prompt_length=2 weight_margin=0.015  warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSITMD task_finetune_irtr_rsitmd_randaug_ViTB32 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 prompt_length=4 weight_margin=0.015  warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSITMD task_finetune_irtr_rsitmd_randaug_ViTB32 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 prompt_length=8 weight_margin=0.015  warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSITMD task_finetune_irtr_rsitmd_randaug_ViTB32 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 prompt_length=16 weight_margin=0.015  warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1


echo ""
echo "****************** RSICD_captions learning_rate=5e-4******************"
echo ""

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSICD_captions task_finetune_irtr_rsicd_randaug_ViTB32 per_gpu_batchsize=$batch_size num_nodes=1 prompt_length=1 weight_margin=0.015 warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSICD_captions task_finetune_irtr_rsicd_randaug_ViTB32 per_gpu_batchsize=$batch_size num_nodes=1 prompt_length=2 weight_margin=0.015 warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSICD_captions task_finetune_irtr_rsicd_randaug_ViTB32 per_gpu_batchsize=$batch_size num_nodes=1 prompt_length=4 weight_margin=0.015 warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSICD_captions task_finetune_irtr_rsicd_randaug_ViTB32 per_gpu_batchsize=$batch_size num_nodes=1 prompt_length=8 weight_margin=0.015 warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSICD_captions task_finetune_irtr_rsicd_randaug_ViTB32 per_gpu_batchsize=$batch_size num_nodes=1 prompt_length=16 weight_margin=0.015 warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1


echo ""
echo "****************** Prompt_length 16  learning_rate=5e-4 ******************"
echo ""
#
CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/UCM_captions task_finetune_irtr_ucm_randaug_ViTB32 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 prompt_length=1 weight_margin=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/UCM_captions task_finetune_irtr_ucm_randaug_ViTB32 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 prompt_length=2 weight_margin=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/UCM_captions task_finetune_irtr_ucm_randaug_ViTB32 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 prompt_length=4 weight_margin=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/UCM_captions task_finetune_irtr_ucm_randaug_ViTB32 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 prompt_length=8 weight_margin=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/UCM_captions task_finetune_irtr_ucm_randaug_ViTB32 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 prompt_length=16 weight_margin=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1


batch_size=96

echo ""
echo "****************** RSITMD_captions learning_rate=5e-4******************"
echo ""



CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSITMD task_finetune_irtr_rsitmd_randaug_ViTB16 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 prompt_length=1 weight_margin=0.015  warmup_steps=0.1 learning_rate=5e-4 loss_scale=1  linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSITMD task_finetune_irtr_rsitmd_randaug_ViTB16 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 prompt_length=2 weight_margin=0.015  warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSITMD task_finetune_irtr_rsitmd_randaug_ViTB16 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 prompt_length=4 weight_margin=0.015  warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSITMD task_finetune_irtr_rsitmd_randaug_ViTB16 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 prompt_length=8 weight_margin=0.015  warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSITMD task_finetune_irtr_rsitmd_randaug_ViTB16 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 prompt_length=16 weight_margin=0.015  warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

echo ""
echo "****************** RSICD_captions learning_rate=5e-4******************"
echo ""

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSICD_captions task_finetune_irtr_rsicd_randaug_ViTB16 per_gpu_batchsize=$batch_size num_nodes=1 prompt_length=1 weight_margin=0.015 warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSICD_captions task_finetune_irtr_rsicd_randaug_ViTB16 per_gpu_batchsize=$batch_size num_nodes=1 prompt_length=2 weight_margin=0.015 warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1
#
CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSICD_captions task_finetune_irtr_rsicd_randaug_ViTB16 per_gpu_batchsize=$batch_size num_nodes=1 prompt_length=4 weight_margin=0.015 warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSICD_captions task_finetune_irtr_rsicd_randaug_ViTB16 per_gpu_batchsize=$batch_size num_nodes=1 prompt_length=8 weight_margin=0.015 warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSICD_captions task_finetune_irtr_rsicd_randaug_ViTB16 per_gpu_batchsize=$batch_size num_nodes=1 prompt_length=16 weight_margin=0.015 warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1


echo ""
echo "****************** UCM ******************"
echo ""

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/UCM_captions task_finetune_irtr_ucm_randaug_ViTB16 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 prompt_length=1 weight_margin=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/UCM_captions task_finetune_irtr_ucm_randaug_ViTB16 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 prompt_length=2 weight_margin=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/UCM_captions task_finetune_irtr_ucm_randaug_ViTB16 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 prompt_length=4 weight_margin=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/UCM_captions task_finetune_irtr_ucm_randaug_ViTB16 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 prompt_length=8 weight_margin=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/UCM_captions task_finetune_irtr_ucm_randaug_ViTB16 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 prompt_length=16 weight_margin=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1





batch_size=36

echo ""
echo "****************** RSITMD_captions learning_rate=5e-4******************"
echo ""



CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSITMD task_finetune_irtr_rsitmd_randaug_ViTL14 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 prompt_length=1 weight_margin=0.015  warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSITMD task_finetune_irtr_rsitmd_randaug_ViTL14 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 prompt_length=2 weight_margin=0.015  warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSITMD task_finetune_irtr_rsitmd_randaug_ViTL14 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 prompt_length=4 weight_margin=0.015  warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSITMD task_finetune_irtr_rsitmd_randaug_ViTL14 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 prompt_length=8 weight_margin=0.015  warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSITMD task_finetune_irtr_rsitmd_randaug_ViTL14 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 prompt_length=16 weight_margin=0.015  warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

echo ""
echo "****************** RSICD_captions learning_rate=5e-4******************"
echo ""

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSICD_captions task_finetune_irtr_rsicd_randaug_ViTL14 per_gpu_batchsize=$batch_size num_nodes=1 prompt_length=1 weight_margin=0.015 warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSICD_captions task_finetune_irtr_rsicd_randaug_ViTL14 per_gpu_batchsize=$batch_size num_nodes=1 prompt_length=2 weight_margin=0.015 warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1
#
CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSICD_captions task_finetune_irtr_rsicd_randaug_ViTL14 per_gpu_batchsize=$batch_size num_nodes=1 prompt_length=4 weight_margin=0.015 warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSICD_captions task_finetune_irtr_rsicd_randaug_ViTL14 per_gpu_batchsize=$batch_size num_nodes=1 prompt_length=8 weight_margin=0.015 warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/RSICD_captions task_finetune_irtr_rsicd_randaug_ViTL14 per_gpu_batchsize=$batch_size num_nodes=1 prompt_length=16 weight_margin=0.015 warmup_steps=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1


echo ""
echo "****************** UCM ******************"
echo ""

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/UCM_captions task_finetune_irtr_ucm_randaug_ViTL14 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 prompt_length=1 weight_margin=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/UCM_captions task_finetune_irtr_ucm_randaug_ViTL14 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 prompt_length=2 weight_margin=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/UCM_captions task_finetune_irtr_ucm_randaug_ViTL14 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 prompt_length=4 weight_margin=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/UCM_captions task_finetune_irtr_ucm_randaug_ViTL14 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 prompt_length=8 weight_margin=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1

CUDA_VISIBLE_DEVICES=$CARD python run_frozen.py with data_root=/data/cross-dataset/dataset/UCM_captions task_finetune_irtr_ucm_randaug_ViTL14 per_gpu_batchsize=$batch_size num_gpus=1 num_nodes=1 precision=32 prompt_length=16 weight_margin=0.1 learning_rate=5e-4 loss_scale=1 linear_projector=0.1


