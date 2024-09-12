import math
import random

import torch
import time
from tqdm import tqdm
import copy as cp
import datetime
import numpy as np
import json
from collections import defaultdict, deque

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import os

from torchvision import transforms
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
from PIL import Image
from prettytable import PrettyTable


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        if len(self.deque) > 0:
            return d.median().item()
        else:
            return d.item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter: SmoothedValue):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()

def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def args_bool_type(x):
    return x.lower() in ['true']


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_multimodal_checkpoint(checkpoint, model, name='model', log=False):
    state_dict = checkpoint[name]
    state_dict_c = cp.copy(state_dict)
    model_state_dict = model.state_dict()

    # remove redundant parameters
    for k, v in state_dict_c.items():
        if 'text_encoder.bert' not in k and \
                'vision_proj.' not in k and 'text_proj.' not in k and 'itm_head' not in k and 'temp' != k:
            del state_dict[k]
            if log:
                print('warning: checkpoint has a redundant parameter: {}, now we remove it'.format(k))

    for k, v in model_state_dict.items():
        if state_dict.get(k, None) is None:
            state_dict[k] = v
            if log:
                print('warning: missing model module: {}, now we use default parameter for it'.format(k))

    return state_dict


def load_pretrain(checkpoint, model, name='model', log=False):
    state_dict = checkpoint[name]
    state_dict['beta_rate'] = model.beta_rate

    for k, v in model.state_dict().items():
        if state_dict.get(k, None) is None:
            state_dict[k] = v
            if log:
                print('warning: missing model module: {}, now we use default parameter for it'.format(k))

    return state_dict


def load_model(checkpoint, model, name='model', log=False):
    state_dict = checkpoint[name]

    for k, v in model.state_dict().items():
        if state_dict.get(k, None) is None:
            state_dict[k] = v
            if log:
                print('warning: missing model module: {}, now we use default parameter for it'.format(k))

    return state_dict


def save_log(cur_output_dir, log: dict, filename="log_info.txt"):
    for k, v in log.items():
        if type(v) == float:
            log[k] = '{:.8f}'.format(v)
    with open(os.path.join(cur_output_dir, filename), "a") as f:
        f.write(json.dumps(log) + "\n")


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def show_attention_relation_v2t(attention_scores, img_path):
    transform_img = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    ])

    attn_spa = int(math.sqrt(attention_scores.shape[-1] - 1))
    a1 = attention_scores.mean(1).mean(1).squeeze(0)[1:].reshape(attn_spa, attn_spa)
    up_sample = nn.Upsample(scale_factor=int(224 / attn_spa), mode='bilinear')
    a1 = up_sample(a1.unsqueeze(0).unsqueeze(0))

    a1 = a1.squeeze(0).squeeze(0)
    a1 = a1.cpu().detach().numpy()

    a2 = attention_scores.mean(1).mean(2).squeeze(0)[1:-1].reshape(1, -1)
    a2 = a2.cpu().detach().numpy()

    image = Image.open(img_path).convert('RGB')
    img = transform_img(image)

    plt.figure()
    plt.imshow(img)
    plt.imshow(a1, alpha=0.5)
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(a2)
    plt.colorbar()
    plt.show()


def show_attention_relation_t2v(attention_scores, img_path):
    transform_img = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    ])

    attn_spa = int(math.sqrt(attention_scores.shape[-2] - 1))
    a1 = attention_scores.mean(1).mean(2).squeeze(0)[1:].reshape(attn_spa, attn_spa)
    up_sample = nn.Upsample(scale_factor=int(224 / attn_spa), mode='bilinear')
    a1 = up_sample(a1.unsqueeze(0).unsqueeze(0))

    a1 = a1.squeeze(0).squeeze(0)
    a1 = a1.cpu().detach().numpy()

    a2 = attention_scores.mean(1).mean(1).squeeze(0)[1:-1].reshape(1, -1)
    a2 = a2.cpu().detach().numpy()

    image = Image.open(img_path).convert('RGB')
    img = transform_img(image)

    plt.figure()
    plt.imshow(img)
    plt.imshow(a1, alpha=0.5)
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(a2)
    plt.colorbar()
    plt.show()


def show_attention_relation_mv(attention_scores, img_path, dim=-1):
    transform_img = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    ])

    attn_spa = int(math.sqrt(attention_scores.shape[dim]))
    a1 = attention_scores.mean(-dim).squeeze(0).reshape(attn_spa, attn_spa)
    up_sample = nn.Upsample(scale_factor=int(224 / attn_spa), mode='bilinear')
    a1 = up_sample(a1.unsqueeze(0).unsqueeze(0))

    a1 = a1.squeeze(0).squeeze(0)
    a1 = a1.cpu().detach().numpy()

    image = Image.open(img_path).convert('RGB')
    img = transform_img(image)

    plt.figure()
    plt.imshow(img)
    plt.imshow(a1, alpha=0.5)
    plt.colorbar()
    plt.show()








