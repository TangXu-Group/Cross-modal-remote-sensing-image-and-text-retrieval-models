# import wandb
import torch
import os
import json
from tqdm import tqdm
from copy import deepcopy
from torch import nn
import copy as cp
import torch.nn.functional as F
from lpe.scheduler import adjust_learning_rate
from .utils import MetricLogger, SmoothedValue, save_log
from lpe.modules.lpe_utils import Tmp_input, alter_idx
from ..modules.multimodal_model_pretrain import momentum_update


def train_backbone(args, model, data_loader, tokenizer, optimizer, epoch, device, text_splitor,
                   cur_output_dir):
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('loss_itm', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itc', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mlm', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mim', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (image, text, mask, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # # we use a per iteration lr scheduler
        adjust_learning_rate(optimizer, epoch + float(i) / len(data_loader), args, 'model')

        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)

        idx_mask = alter_idx(
            args=args,
            text=text, idx=idx,
            txt2id_dict=data_loader.dataset.txt2id_dict,
            txt2img=data_loader.dataset.txt2img
        )

        text_input = tokenizer(text, padding='longest', max_length=30, return_tensors="pt").to(device)

        # if args.use_alter_mlm:
        #     noun_list = []
        #     noun_idxs = []
        #     for txt in text:
        #         text_doc = text_splitor(txt)
        #         noun_list.append([token.lemma_ for token in text_doc if token.pos_ == "NOUN"])
        #         noun_idxs.append([token.i + 1 for token in text_doc if token.pos_ == "NOUN"])   # the input of bert model contain the head 'cls'
        # else:
        #     noun_idxs = None

        if args.use_alter_mlm:
            noun_list = []
            noun_idxs = []
            for txt in text:
                text_doc = text_splitor(txt)
                list_idxs = []
                list_noun = []
                for token in text_doc:
                    if token.is_stop is not True or token.pos_ == 'NUM':
                        list_idxs.append(token.i + 1)       # the input of bert contain a head 'cls'
                        list_noun.append(token.text)
                noun_idxs.append(list_idxs)
                noun_list.append(list_noun)
        else:
            noun_idxs = None

        loss_itc, loss_itm, loss_mlm, loss_mim, image_idxs_neg, text_idxs_neg = model(
            image, text_input, idx_mask=idx_mask, noun_idxs=noun_idxs, img_mask=mask)

        if loss_itm is None:
            loss = loss_itc
            loss_itm = torch.tensor(0.)
            loss_mlm = torch.tensor(0.)
            loss_mim = torch.tensor(0.)
        else:
            loss = loss_itc + loss_itm + loss_mlm + loss_mim

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log = {
            'loss_itm': loss_itm.item(),
            'loss_itc': loss_itc.item(),
            'loss_mlm': loss_mlm.item(),
            'loss_mim': loss_mim.item(),
            'loss': loss.item(),
            'lr': optimizer.param_groups[0]["lr"]
        }

        metric_logger.update(**log)
        save_log(cur_output_dir=cur_output_dir, log=log)

        if args.wandb_log:
            wandb.log({
                'loss_itm': loss_itm.item(),
                'loss_itc': loss_itc.item(),
                "lr": optimizer.param_groups[0]["lr"]
            }, step=args.step)
            args.step += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def train_backbone_control_group(args, model, data_loader, tokenizer, optimizer, epoch, warmup_steps, device, cur_output_dir):
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('loss_itm', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itc', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mlm', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mim', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    for i, (image, text, mask, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # # we use a per iteration lr scheduler
        adjust_learning_rate(optimizer, epoch + float(i) / len(data_loader), args, 'model')

        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        text_input = tokenizer(text, padding='longest', max_length=30, return_tensors="pt").to(device)

        loss_itc, vl_output, itm_labels, loss_mlm, loss_mim, _, _, _, _ = model(image, text_input, idx=idx, img_mask=mask)
        loss_itm = F.cross_entropy(vl_output, itm_labels.squeeze(1))
        loss = loss_itc + loss_itm + loss_mlm + loss_mim

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log = {
            'loss_itm': loss_itm.item(),
            'loss_itc': loss_itc.item(),
            'loss_mlm': loss_mlm.item(),
            'loss_mim': loss_mim.item(),
            'lr': optimizer.param_groups[0]["lr"]
        }

        metric_logger.update(**log)
        save_log(cur_output_dir=cur_output_dir, log=log)

        if args.wandb_log:
            wandb.log({
                'loss_itm': loss_itm.item(),
                'loss_itc': loss_itc.item(),
                "lr": optimizer.param_groups[0]["lr"]
            }, step=args.step)
            args.step += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def train_lpe(args, model, model_m, data_loader, tokenizer, optimizer, text_splitor,
              epoch, device, cur_output_dir):
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('beta', SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('weight', SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('beta_rate', SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('beta_loss', SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('no_alter_num', SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('loss_itm', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('alter_loss_itm', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itc', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mlm', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mim', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('exp_q_size', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (image, text, mask, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # # we use a per iteration lr scheduler
        adjust_learning_rate(optimizer, epoch + float(i) / len(data_loader), args, 'model')

        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)

        idx_mask = alter_idx(
            args=args,
            text=text, idx=idx,
            txt2id_dict=data_loader.dataset.txt2id_dict,
            txt2img=data_loader.dataset.txt2img,
        )

        text_input = tokenizer(text, padding='longest', max_length=30, return_tensors="pt").to(device)

        if args.use_alter_mlm:
            noun_list = []
            noun_idxs = []
            for txt in text:
                text_doc = text_splitor(txt)
                list_idxs = []
                list_noun = []
                for token in text_doc:
                    if token.is_stop is not True or token.pos_ == 'NUM':
                        list_idxs.append(token.i + 1)       # the input of bert contain a head 'cls'
                        list_noun.append(token.text)
                noun_idxs.append(list_idxs)
                noun_list.append(list_noun)
        else:
            noun_idxs = None

        loss_itm, alter_loss_itm, loss_itc, loss_mlm, loss_mim, beta_loss, beta, weight, no_alter_num = model(
            image, text_input, idx_mask=idx_mask, noun_idxs=noun_idxs, img_mask=mask, orgin_text=text,
            epoch=epoch, model_m=model_m, use_experience=args.use_experience)

        loss = loss_itc + alter_loss_itm + loss_mlm + loss_mim + beta_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.use_experience:
            momentum_update(model=model, model_m=model_m, momentum_coeff=args.momentum)
        if args.distributed:
            exp_q_size = model.module.lpe_buffer.get_exp_queue_size()
            beta_rate = model.module.beta_rate
        else:
            exp_q_size = model.lpe_buffer.get_exp_queue_size()
            beta_rate = model.beta_rate

        log = {
            'beta': beta.item(),
            'weight': weight.item(),
            'beta_rate': beta_rate.item(),
            'beta_loss': beta_loss.item(),
            'no_alter_num': no_alter_num,
            'loss_itm': loss_itm.item(),
            'alter_loss_itm': alter_loss_itm.item(),
            'loss_itc': loss_itc.item(),
            'loss_mlm': loss_mlm.item(),
            'loss_mim': loss_mim.item(),
            'lr': optimizer.param_groups[0]["lr"],
            'exp_q_size': exp_q_size
        }

        metric_logger.update(**log)
        save_log(cur_output_dir=cur_output_dir, log=log)

        if args.wandb_log:
            wandb.log({
                'loss_itm': loss_itm.item(),
                'loss_itc': loss_itc.item(),
                "lr": optimizer.param_groups[0]["lr"]
            }, step=args.step)
            args.step += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


# def calculate_alter_itm_loss(args, lpe_buffer, vl_output, itm_labels, vl_embeddings_m, bs, epoch, model_m, score):
#     itm_targets = torch.cat([1 - itm_labels, itm_labels], dim=1)
#
#     neg_alter_label = torch.clip(itm_labels - 1, max=0)
#     itm_alter_label = itm_labels + neg_alter_label
#
#     exp_itm_label, gamma = lpe_buffer.get_exp_targets(
#         bs=bs, epoch=epoch,
#         itm_alter_label=itm_alter_label,
#         model_m=model_m, score=score
#     )
#     beta = gamma * args.beta
#
#     exp_itm_target = torch.cat([1 - exp_itm_label, exp_itm_label], dim=1)
#     alter_itm_target = beta * exp_itm_target + (1 - beta) * itm_targets
#     norm_factor = torch.sum(alter_itm_target, dim=1).unsqueeze(1)
#     alter_itm_target /= torch.cat([norm_factor, norm_factor], dim=1)
#
#     push_embedding = vl_embeddings_m
#     push_label = itm_alter_label
#
#     lpe_buffer.exp_dequeue_and_enqueue(vl_feat=push_embedding, label=push_label)
#
#     alter_loss_itm = -torch.sum(F.log_softmax(vl_output, dim=1) * alter_itm_target, dim=1).mean()
#
#     return alter_loss_itm, beta

