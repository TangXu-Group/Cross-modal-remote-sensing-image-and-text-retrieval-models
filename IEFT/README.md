# IEFT : Interacting-Enhancing Feature Transformer for Cross-modal Remote Sensing Image and Text Retrieval


## Install

Please follow requirements.txt



## Download Pretrained Weights

We leverage the pretrained weight from ViLT as ICML 2021 (long talk) paper: "[ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334)" 

ViLT-B/32 Pretrained with MLM+ITM for 200k steps on GCC+SBU+COCO+VG (ViLT-B/32 200k) [this link](https://github.com/dandelin/ViLT/releases/download/200k/vilt_200k_mlm_itm.ckpt)

# Dataset Preparation

We follow [ViLT](https://github.com/dandelin/ViLT) and use `pyarrow` to serialize the datasets. See [this link](https://github.com/dandelin/ViLT/blob/master/DATA.md) for details.

# Training Stage

python run.py with data_root=<ARROW_ROOT> task_finetune_irtr_sydney_randaug per_gpu_batchsize=<BS_FITS_YOUR_GPU> num_gpus=1 num_nodes=1 precision=<PRECISION, as 32 or 16> load_path="/x/vilt_200k_mlm_itm.ckpt"

python run.py with data_root=<ARROW_ROOT> task_finetune_irtr_ucm_randaug per_gpu_batchsize=<BS_FITS_YOUR_GPU> num_gpus=1 num_nodes=1 precision=<PRECISION, as 32 or 16> load_path="/x/vilt_200k_mlm_itm.ckpt"

python run.py with data_root=<ARROW_ROOT> task_finetune_irtr_rsitmd_randaug per_gpu_batchsize=<BS_FITS_YOUR_GPU> num_gpus=1 num_nodes=1 precision=<PRECISION, as 32 or 16> load_path="/x/vilt_200k_mlm_itm.ckpt"

python run.py with data_root=<ARROW_ROOT> task_finetune_irtr_rsicd_randaug per_gpu_batchsize=<BS_FITS_YOUR_GPU> num_gpus=1 num_nodes=1 precision=<PRECISION, as 32 or 16> load_path="/x/vilt_200k_mlm_itm.ckpt"


# Testing Stage
python run.py with data_root=<ARROW_ROOT> task_finetune_irtr_sydney_randaug per_gpu_batchsize=<BS_FITS_YOUR_GPU> num_gpus=1 num_nodes=1 precision=<PRECISION, as 32 or 16> test_only=True load_path=<TRAINED_MODEL>

python run.py with data_root=<ARROW_ROOT> task_finetune_irtr_ucm_randaug per_gpu_batchsize=<BS_FITS_YOUR_GPU> num_gpus=1 num_nodes=1 precision=<PRECISION, as 32 or 16> test_only=True load_path=<TRAINED_MODEL>

python run.py with data_root=<ARROW_ROOT> task_finetune_irtr_rsitmd_randaug per_gpu_batchsize=<BS_FITS_YOUR_GPU> num_gpus=1 num_nodes=1 precision=<PRECISION, as 32 or 16> test_only=True load_path=<TRAINED_MODEL>

python run.py with data_root=<ARROW_ROOT> task_finetune_irtr_rsicd_randaug per_gpu_batchsize=<BS_FITS_YOUR_GPU> num_gpus=1 num_nodes=1 precision=<PRECISION, as 32 or 16> test_only=True load_path=<TRAINED_MODEL>

The returned values are IR R@1, R@5, R@10 and TR R@1, R@5, R@10. (Only one gpu) 


## Contact for Issues
- [Yijing_Wang](1016676609@qq.com)
