import argparse

import spacy
import torch
import time
import datetime
import os
import json
import wandb
import random
import numpy as np
from pathlib import Path
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer

from lpe.configs.swin_transformer_config import get_config
from lpe.datasets import create_datasets, create_loader, create_sampler
from lpe.modules.multimodal_model_pretrain import CRSRT

from lpe.optim import create_optimizer
from lpe.scheduler import create_scheduler
from lpe.utils.utils import load_model, set_seed, args_bool_type, load_multimodal_checkpoint, count_parameters
from lpe.utils.train import train_backbone, train_backbone_control_group
from lpe.utils.evaluate import evaluation, evaluation_contrastive_only, itm_eval
from lpe.utils import utils

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_args_parser():
    parser = argparse.ArgumentParser('LPE for image-text retrieval', add_help=False)

    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch (use in resuming checkpoint)')
    parser.add_argument('--save_epoch', default=5, type=int)
    parser.add_argument('--output_dir', default=r'/data1/amax/hdb/output-5.4.1', type=str)
    parser.add_argument('--log_dir', default=r'/data1/amax/hdb/output-5.4.1', type=str)
    parser.add_argument('--cfg',
                        default=r'/home/amax/hdb/LPE-master-5.4.1/lpe/configs/swin_small_patch4_window7_224.yaml')
    parser.add_argument('--pretrain',
                        default='',
                        type=str, help='resume from checkpoint')
    parser.add_argument('--resume',
                        default=None,
                        type=str, help='resume from checkpoint')

    # ======================== data parameter =======================================

    # RSITMD
    parser.add_argument('--data_name', default='RSITMD', type= str, help='dataset name')
    parser.add_argument('--data_dir', default=r'/data1/amax/datasets/RSITMD', type=str,
                        help='dataset dir')
    parser.add_argument('--image_dir', default=r'/data1/amax/datasets/RSITMD/root/images', type=str,
                        help='image dir')
    parser.add_argument('--train_file', default=['json_root/rsitmd_train.json'], nargs='+',
                        help='list of train file name')
    parser.add_argument('--val_file', default='json_root/rsitmd_test.json', type=str,
                        help='val file name')
    parser.add_argument('--test_file', default='json_root/rsitmd_test.json', type=str,
                        help='test file name')

    # # RSICD
    # parser.add_argument('--data_name', default='RSICD', type= str, help='dataset name')
    # parser.add_argument('--data_dir', default=r'/data1/amax/datasets/RSICD', type=str,
    #                     help='dataset dir')
    # parser.add_argument('--image_dir', default=r'/data1/amax/datasets/RSICD/root/RSICD_images', type=str,
    #                     help='image dir')
    # parser.add_argument('--train_file', default=['json_root/rsicd_train.json'], nargs='+',
    #                     help='list of train file name')
    # parser.add_argument('--val_file', default='json_root/rsicd_val.json', type=str,
    #                     help='val file name')
    # parser.add_argument('--test_file', default='json_root/rsicd_test.json', type=str,
    #                     help='test file name')

    # UCM
    # parser.add_argument('--data_name', default='UCM', type= str, help='dataset name')
    # parser.add_argument('--data_dir', default=r'/data1/amax/datasets/UCM_captions', type=str,
    #                     help='dataset dir')
    # parser.add_argument('--image_dir', default=r'/data1/amax/datasets/UCM_captions/root/imgs', type=str,
    #                     help='image dir')
    # parser.add_argument('--train_file', default=['json_root/ucm_train.json'], nargs='+',
    #                     help='list of train file name')
    # parser.add_argument('--val_file', default='json_root/ucm_val.json', type=str,
    #                     help='val file name')
    # parser.add_argument('--test_file', default='json_root/ucm_test.json', type=str,
    #                     help='test file name')

    # ========================= model parameters ====================================
    # model
    parser.add_argument('--text_encoder', default='/data1/amax/pretrain_model/bert-base-uncased')
    parser.add_argument('--visual_encoder',
                        default='/data1/amax/pretrain_model/swin-transformer/swin-s/swin_small_patch4_window7_224.pth')
    parser.add_argument('--beit_encoder', default='/data1/amax/pretrain_model/beit_base_patch16_224_pt22k')
    parser.add_argument('--modify_bert_config', default='/home/amax/hdb/LPE-master-5.4.1/lpe/configs')
    parser.add_argument('--modify_beit_config', default='/home/amax/hdb/LPE-master-5.4.1/lpe/configs')

    parser.add_argument('--use_multiscale_visual_fusion', type=args_bool_type, default=True)
    parser.add_argument('--high_level_dim', type=int, default=768)
    parser.add_argument('--low_level_dim', type=int, default=192)
    parser.add_argument('--low_level_dim_list', type=list, default=[96, 96, 192, 384])
    parser.add_argument('--num_low_level_use', type=int, default=4)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--encoder_stride', type=int, default=32)
    parser.add_argument('--mlm_probability', type=float, default=0.15)
    parser.add_argument('--use_mm_fusion', type=args_bool_type, default=True)
    parser.add_argument('--use_mlm', type=args_bool_type, default=True)
    parser.add_argument('--use_mim', type=args_bool_type, default=True)
    parser.add_argument('--use_alter_mlm', type=args_bool_type, default=True)
    parser.add_argument('--use_mm_fusion_query', type=args_bool_type, default=True)
    parser.add_argument('--temp', type=float, default=0.07)
    parser.add_argument('--momentum', type=float, default=0.995)
    parser.add_argument('--k_test', type=int, default=256)  # 256

    # experience queue
    parser.add_argument('--beta', default=1, type=float)
    parser.add_argument('--cfd_epoch', default=65, type=int,
                        help='the confidence epoch, when epoch is larger than it, the experiences in the buffer will be fully accept')
    parser.add_argument('--start_exp_epoch', type=int, default=0,
                        help='the epoch when model uses LPE module')
    parser.add_argument('--use_experience', type=args_bool_type, default=False)
    parser.add_argument('--is_use_experience', type=args_bool_type, default=True)
    parser.add_argument('--pre_filling', type=args_bool_type, default=False)
    parser.add_argument('--exp_queue_capacity', type=int, default=65536)
    parser.add_argument('--dist_threshold', type=float, default=0.08)
    parser.add_argument('--sim_threshold', type=float, default=0.866)  # sqrt(3) / 2

    # ======================== train parameter ======================================
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--ngpus', default=1, type=int, help='number of GPUs to request on each node')
    parser.add_argument('--nodes', default=1, type=int, help='Number of nodes to request')
    parser.add_argument('--evaluate', default=False, type=args_bool_type, help='if only evaluate model')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)

    parser.add_argument('--blr', default=8e-5, type=float, help='base learning rate for linear scaling rule')
    parser.add_argument('--lr', default=None, type=float, help='lr(absolute_lr) = blr * total_batch_size / 256')
    parser.add_argument('--batch_size_train', default=32, type=int)
    parser.add_argument('--batch_size_test', default=64, type=int)
    parser.add_argument('--min_lr', type=float, default=1e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    # optimizer
    parser.add_argument('--opt', default='adamW')
    parser.add_argument('--lr_decay', default=1, type=float, help='finetune lr decay')
    parser.add_argument('--lr_decay_disc', default=1, type=float, help='lr decay for discriminator')
    parser.add_argument('--weight_decay', default=0.02, type=float)

    # scheduler
    parser.add_argument('--sched', default='cosine')
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--decay_rate', default=1)
    parser.add_argument('--warmup_lr', default=1e-5)
    parser.add_argument('--warmup_epochs', default=5, type=int, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--cooldown_epochs', default=0, type=int)

    # ========================= wandb config ======================================
    parser.add_argument('--wandb_log', default=False, type=bool, help='whether use wandb or not')
    parser.add_argument('--wandb_id', default=None, type=str)
    parser.add_argument('--project', default='LPE for image-text retrieval', type=str)
    parser.add_argument('--entity', default='xdu_hdb', type=str)
    parser.add_argument('--experiment', default='CRSRT_test', type=str,
                        help='1. CRSRT'
                             '2. CRSRT_only_hlfn'
                             '3. CRSRT_only_hlfn+mvit'
                             '4. CRSRT_directly'
                             '5. CRSRT_from_scratch'
                        )
    parser.add_argument('--step', default=0, type=int)

    return parser.parse_args()


def main(args):
    print(args)
    swin_cfg = get_config(args)
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    set_seed(args.seed)

    print("successfully load data.")

    # load model
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    backbone = CRSRT(args=args, swin_cfg=swin_cfg, tokenizer=tokenizer)

    txt_config = backbone.text_encoder.config
    img_config = backbone.visual_fusion_net.visual_fusion_encoder.config

    txt_use_layer = list(range(txt_config.text_start, txt_config.text_end)) + \
                    list(range(txt_config.fusion_start, txt_config.fusion_end))
    img_use_layer = list(range(img_config.fusion_start, img_config.fusion_end))

    txt_not_use = [i for i in range(txt_config.num_hidden_layers) if i not in txt_use_layer]
    img_not_use = [i for i in range(img_config.num_hidden_layers) if i not in img_use_layer]

    for idx in txt_not_use:
        for param in backbone.text_encoder.bert.encoder.layer[idx].parameters():
            param.requires_grad = False

    for idx in img_not_use:
        for param in backbone.visual_fusion_net.visual_fusion_encoder.encoder.layer[idx].parameters():
            param.requires_grad = False

    count_parameters(backbone)


if __name__ == '__main__':
    args = get_args_parser()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

