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
import torch.distributed as dist

from lpe.configs.swin_transformer_config import get_config
from lpe.datasets import create_datasets, create_loader, create_sampler
from lpe.modules.lpe_module import LPEBuffer
from lpe.modules.multimodal_model_irtr import CRSRT
from lpe.optim import create_optimizer
from lpe.scheduler import create_scheduler
from lpe.utils.utils import load_model, args_bool_type, set_seed, load_multimodal_checkpoint, load_pretrain, \
    count_parameters
from lpe.utils.train import train_lpe, train_backbone
from lpe.utils.evaluate import evaluation, itm_eval
from lpe.utils import utils

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_args_parser():
    parser = argparse.ArgumentParser('LPE for image-text retrieval', add_help=False)

    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch (use in resuming checkpoint)')
    parser.add_argument('--save_epoch', default=5, type=int)
    parser.add_argument('--output_dir', default=r'/data1/amax/hdb/Retrieval/output-5.4.1-param-exp', type=str)
    parser.add_argument('--log_dir', default=r'/data1/amax/hdb/Retrieval/output-5.4.1-param-exp', type=str)
    parser.add_argument('--cfg',
                        default=r'/home/amax/hdb/Retrieval/LPE-master-5.4.1/lpe/configs/swin_small_patch4_window7_224.yaml')
    parser.add_argument('--pretrain',
                        default=None,
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

    # # UCM
    # parser.add_argument('--data_name', default='UCM', type= str, help='dataset name')
    # parser.add_argument('--data_dir', default=r'/data1/amax/datasets/UCM_captions', type=str,
    #                     help='dataset dir')
    # parser.add_argument('--image_dir', default=r'/data1/amax/datasets/UCM_captions/root/imgs', type=str,
    #                     help='image dir')
    # parser.add_argument('--train_file', default=['json_root/ucm_train.json'], nargs='+',
    #                     help='list of train file name')
    # parser.add_argument('--val_file', default='json_root/ucm_test.json', type=str,
    #                     help='val file name')
    # parser.add_argument('--test_file', default='json_root/ucm_test.json', type=str,
    #                     help='test file name')


    # ========================= model parameters ====================================
    # model
    parser.add_argument('--text_encoder', default='/data1/amax/pretrain_model/bert-base-uncased')
    parser.add_argument('--visual_encoder',
                        default='/data1/amax/pretrain_model/swin-transformer/swin-s/swin_small_patch4_window7_224.pth')
    parser.add_argument('--beit_encoder', default='/data1/amax/pretrain_model/beit_base_patch16_224_pt22k')
    parser.add_argument('--modify_bert_config', default='/home/amax/hdb/Retrieval/LPE-master-5.4.1/lpe/configs')
    parser.add_argument('--modify_beit_config', default='/home/amax/hdb/Retrieval/LPE-master-5.4.1/lpe/configs')

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
    parser.add_argument('--use_mm_fusion_query', type=args_bool_type, default=False)
    parser.add_argument('--temp', type=float, default=0.07)
    parser.add_argument('--momentum', type=float, default=0.995)
    parser.add_argument('--k_test', type=int, default=256)  # 256

    # experience queue
    parser.add_argument('--beta', default=1, type=float)
    parser.add_argument('--beta_rate', default=0.05, type=float)    # 0.05
    parser.add_argument('--cfd_epoch', default=65, type=int,
                        help='the confidence epoch, when epoch is larger than it, the experiences in the buffer will be fully accept')
    parser.add_argument('--start_exp_epoch', type=int, default=0,
                        help='the epoch when model uses LPE module')
    parser.add_argument('--use_experience', type=args_bool_type, default=False)
    parser.add_argument('--is_use_experience', type=args_bool_type, default=True)
    parser.add_argument('--pre_filling', type=args_bool_type, default=False)
    parser.add_argument('--exp_queue_capacity', type=int, default=65536)
    parser.add_argument('--dist_threshold', type=float, default=0.08)
    parser.add_argument('--sim_threshold', type=float, default=0.866)   # sqrt(3) / 2

    # ======================== train parameter ======================================
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--ngpus', default=1, type=int, help='number of GPUs to request on each node')
    parser.add_argument('--nodes', default=1, type=int, help='Number of nodes to request')
    parser.add_argument('--evaluate', default=False, type=args_bool_type, help='if only evaluate model')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--distributed', default=False, type=args_bool_type)

    parser.add_argument('--blr', default=8e-5, type=float, help='base learning rate for linear scaling rule')
    parser.add_argument('--lr', default=None, type=float, help='lr(absolute_lr) = blr * total_batch_size / 256')
    parser.add_argument('--batch_size_train', default=64, type=int)
    parser.add_argument('--batch_size_test', default=64, type=int)
    parser.add_argument('--min_lr', type=float, default=1e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    # optimizer
    parser.add_argument('--opt', default='adamW')
    parser.add_argument('--lr_decay', default=0.2, type=float, help='finetune lr decay')
    parser.add_argument('--weight_decay', default=0.02, type=float)

    # scheduler
    parser.add_argument('--sched', default='cosine')
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--decay_rate', default=1)
    parser.add_argument('--warmup_lr', default=1e-5)
    parser.add_argument('--warmup_epochs', default=0, type=int, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--cooldown_epochs', default=0, type=int)

    # ========================= wandb config ======================================
    parser.add_argument('--wandb_log', default=False, type=args_bool_type, help='whether use wandb or not')
    parser.add_argument('--wandb_id', default=None, type=str)
    parser.add_argument('--project', default='LPE for image-text retrieval', type=str)
    parser.add_argument('--entity', default='xdu_hdb', type=str)
    parser.add_argument('--experiment', default='CRSRT_LPE_test', type=str,
                        help='1. CRSRT_LPE_scratch'
                        )
    parser.add_argument('--step', default=0, type=int)

    return parser.parse_args()


def wandb_init(args):
    if args.wandb_id is None:
        args.wandb_id = wandb.util.generate_id()
    print('wandb id: {}'.format(args.wandb_id))

    config = {
        "device": "NVIDIA RTX A6000",
        "experiment": args.experiment,
        "blr": args.blr,
        "lr": args.lr,
        "batch_size": args.batch_size_train,
        "ngpus": args.ngpus,
        "nodes": args.nodes,
        "sched": args.sched
    }
    wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.experiment + "(id={})".format(args.wandb_id),
        config=config,
        notes="LPE",
        id=args.wandb_id,
        resume='allow'
    )


def main(args):
    print(args)
    swin_cfg = get_config(args)
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    set_seed(args.seed)

    # load datasets
    train_dataset, val_dataset, test_dataset = create_datasets(args, swin_cfg)
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                          batch_size=[args.batch_size_train] + [
                                                              args.batch_size_test] * 2,
                                                          num_workers=[4, 4, 4],
                                                          is_trains=[True, False, False],
                                                          collate_fns=[None, None, None])
    print("successfully load data.")

    if args.exp_queue_capacity > len(train_loader) * args.batch_size_train * 5 * utils.get_world_size():
        args.exp_queue_capacity = len(train_loader) * args.batch_size_train * 5 * utils.get_world_size()

    total_batch_size = args.batch_size_train * utils.get_world_size()
    args.lr = args.blr * total_batch_size / 256

    # load model
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    text_splitor = spacy.load("en_core_web_sm")

    model = CRSRT(args=args, swin_cfg=swin_cfg, tokenizer=tokenizer)
    model_m = CRSRT(args=args, swin_cfg=swin_cfg, tokenizer=tokenizer)

    if args.pretrain:
        checkpoint = torch.load(args.pretrain, map_location='cpu')
        state_dict = load_pretrain(checkpoint, model)
        model.load_state_dict(state_dict)

        args.start_epoch = checkpoint['epoch'] + 1
        best = checkpoint['best']
        best_epoch = checkpoint['best_epoch']
        args.step = checkpoint['step']

        print('load pretrain checkpoint from %s' % args.pretrain)
        print('start epoch: ', args.start_epoch)

    # load resume checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        state_dict = load_model(checkpoint, model)
        state_dict_m = load_model(checkpoint, model, 'model_m')
        # state_dict_critic = load_critic(checkpoint, critic)

        model.load_state_dict(state_dict)
        model_m.state_dict(state_dict_m)
        # lpe_buffer.load_state_dict(buffer_dict)
        # critic.load_state_dict(state_dict_critic)

        args.start_epoch = checkpoint['epoch'] + 1
        best = checkpoint['best']
        best_epoch = checkpoint['best_epoch']
        args.step = checkpoint['step']

        args.is_use_experience = checkpoint['args'].is_use_experience

        print('load checkpoint from %s' % args.resume)
        print('resume epoch: ', args.start_epoch)
    else:
        best = 0; best_epoch = 0

    model.to(device)
    model_m.to(device)

    model_m.load_state_dict(model.state_dict())
    print('momentum model successfully load online parameter')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    model_without_ddp_best = CRSRT(args=args, swin_cfg=swin_cfg, tokenizer=tokenizer).to(device)
    model_without_ddp_best.load_state_dict(model_without_ddp.state_dict())

    print('dataset: {}'.format(args.data_name))
    print('base lr: {}'.format(args.blr))
    print('total batch size: {}'.format(total_batch_size))
    print('absolutely lr: {}'.format(args.lr))
    print('beta: {}'.format(args.beta))
    print('dist_threshold: {}'.format(args.dist_threshold))
    print('exp queue capacity: {}'.format(args.exp_queue_capacity))

    optimizer = create_optimizer(args, [model], ['model'], is_distributed=args.distributed)
    # disc_opt = create_optimizer(args, [encoder, discriminator], ['encoder', 'discriminator'])
    # critic_opt = create_optimizer(args, [critic], ['critic'])

    max_epoch = args.epochs
    warmup_steps = args.warmup_epochs

    if args.evaluate:
        args.start_epoch = 0
        max_epoch = 1
        args.pre_filling = False
    elif args.wandb_log:
        wandb_init(args)

    cur_output_dir = os.path.join(args.output_dir,
                                  "{}/{}_blr_{:.8f}_beta_{}_q_capacity_{}_exp_{}_dist_theta_{}_beta_rate_{}_num_low_level_use_{}_sim_threshold_{}".format(
                                      args.data_name, args.experiment, args.blr, args.beta, args.exp_queue_capacity,
                                      args.is_use_experience, args.dist_threshold, args.beta_rate, args.num_low_level_use, args.sim_threshold
                                  ))
    os.makedirs(cur_output_dir, exist_ok=True)

    print('experiment name: ', args.experiment)

    # if args.pre_filling:
    #     print('--------------------------------------------------------------------')
    #     print('start pre-filling')
    #     print('--------------------------------------------------------------------')
    #     pre_filling(args=args, discriminator=discriminator, generator=generator, encoder=encoder, disc_opt=disc_opt,
    #                 data_loader=train_loader, tokenizer=tokenizer, cur_output_dir=cur_output_dir,
    #                 experience_queue=experience_queue, device=device)

    print('--------------------------------------------------------------------')
    print('start training')
    print('start epoch: {} | max epoch: {}'.format(args.start_epoch, max_epoch))
    print('--------------------------------------------------------------------')
    start_time = time.time()
    for epoch in range(args.start_epoch, max_epoch):
        if not args.evaluate:
            if epoch + 1 >= args.start_exp_epoch:
                args.use_experience = args.is_use_experience
            else:
                model_m.load_state_dict(model_without_ddp.state_dict())
            # if epoch + 1 < args.start_exp_epoch or (epoch + 1 >= args.start_exp_epoch and epoch + 1 % 10 == 0):
            #     model_m.load_state_dict(model_without_ddp.state_dict())
            if epoch + 1 >= args.start_exp_epoch:
                # args.use_experience = args.is_use_experience
                train_stats = train_lpe(args=args, model=model, model_m=model_m, data_loader=train_loader,
                                        tokenizer=tokenizer,
                                        optimizer=optimizer, text_splitor=text_splitor,
                                        epoch=epoch, device=device,
                                        cur_output_dir=cur_output_dir)
            else:
                train_stats = train_backbone(args=args, model=model, data_loader=train_loader,
                                             tokenizer=tokenizer,
                                             optimizer=optimizer,
                                             epoch=epoch, device=device,
                                             text_splitor=text_splitor,
                                             cur_output_dir=cur_output_dir)
                model_m.load_state_dict(model_without_ddp.state_dict())

        score_val_i2t, score_val_t2i = evaluation(args, model_without_ddp, val_loader, tokenizer, device)

        if utils.is_main_process():
            val_result = itm_eval(score_val_i2t, score_val_t2i,
                                  val_loader.dataset.txt2img, val_loader.dataset.img2txt,
                                  val_loader.dataset.txt2id_dict, val_loader.dataset.id2txt_dict,
                                  val_loader.dataset.text)

            print('val result: {}'.format(val_result))
            print('best epoch: {}'.format(best_epoch))

            if val_result['r_mean'] > best:
                best = val_result['r_mean']
                best_epoch = epoch
                model_without_ddp_best.load_state_dict(model_without_ddp.state_dict())
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'model_m': model_m.state_dict(),
                    'args': args,
                    'epoch': epoch,
                    'best_epoch': best_epoch,
                    'best': best,
                    'step': args.step
                }
                torch.save(save_obj, os.path.join(cur_output_dir, 'checkpoint_best.pth'))
                print('save best model')

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_result.items()},
                         'epoch': epoch,
                         'best epoch': best_epoch
                         }
            # log_stats = {**{f'val_{k}': v for k, v in val_result.items()},
            #              'epoch': epoch,
            #              'best epoch': best_epoch
            #              }
            if args.wandb_log:
                wandb.log(log_stats, step=args.step)

            with open(os.path.join(cur_output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

            if (epoch + 2) % args.save_epoch == 0:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'model_m': model_m.state_dict(),
                    'args': args,
                    'epoch': epoch,
                    'best_epoch': best_epoch,
                    'best': best,
                    'step': args.step
                }
                save_name = f'checkpoint_epoch_{epoch}_rmean_{val_result["r_mean"]}.pth'
                torch.save(save_obj, os.path.join(cur_output_dir, save_name))
                print('save model')

            save_obj = {
                'model': model_without_ddp.state_dict(),
                'model_m': model_m.state_dict(),
                'args': args,
                'epoch': epoch,
                'best_epoch': best_epoch,
                'best': best,
                'step': args.step
            }
            torch.save(save_obj, os.path.join(cur_output_dir, 'current_checkpoint.pth'))

        if args.distributed:
            dist.barrier()
        torch.cuda.empty_cache()

    if not args.evaluate:
        save_obj = {
            'model': model_without_ddp.state_dict(),
            'model_m': model_m.state_dict(),
            'args': args,
            'epoch': max_epoch,
            'best_epoch': best_epoch,
            'best': best,
            'step': args.step
        }
        save_name = f'checkpoint_last.pth'
        torch.save(save_obj, os.path.join(cur_output_dir, save_name))
        print('save model')

    # print('testing model...')
    # score_test_i2t, score_test_t2i = evaluation(args, model_without_ddp_best, test_loader, tokenizer, device)
    # test_result = itm_eval(score_test_i2t, score_test_t2i,
    #                        test_loader.dataset.txt2img, test_loader.dataset.img2txt,
    #                        test_loader.dataset.txt2id_dict, test_loader.dataset.id2txt_dict,
    #                        test_loader.dataset.text)
    # print('test result: {}'.format(test_result))
    # log_stats = {**{f'test_{k}': v for k, v in test_result.items()},
    #              'best epoch': best_epoch,
    #              }
    # with open(os.path.join(cur_output_dir, "log.txt"), "a") as f:
    #     f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    print('best epoch: %d' % best_epoch)


if __name__ == '__main__':
    args = get_args_parser()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # args.device = 'cuda:1'
    main(args)
    # for beta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    #     args.beta = beta
    #     for coeff in [1e-1, 1e-3, 1e-5]:
    #         args.balance_coeff = coeff
    #         main(args)
    # for sim_threshold in [0.95, 0.9, 0.85, 0.8, 0.7]:
    #     args.sim_threshold = sim_threshold
    #     main(args)

