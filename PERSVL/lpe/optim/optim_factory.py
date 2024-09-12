""" Optimizer Factory w/ Custom Weight Decay
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import re
from torch import optim as optim

from .adafactor import Adafactor
from .adahessian import Adahessian
from .adamp import AdamP
from .lookahead import Lookahead
from .nadam import Nadam
from .novograd import NovoGrad
from .nvnovograd import NvNovoGrad
from .radam import RAdam
from .rmsprop_tf import RMSpropTF
from .sgdp import SGDP

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def add_decay(model, name, weight_decay=1e-5, lr_decay=0.75, skip_list=[], is_distributed=False):
    """add weight decay and lr decay"""
    param_group_names = {}
    param_groups = {}

    if is_distributed:
        txt_config = model.module.text_encoder.config
    else:
        txt_config = model.text_encoder.config
    txt_fusion_layer = list(range(txt_config.fusion_start, txt_config.fusion_end))
    layerExtract = '.+\..+\..+\..+\.(\d+)\..*'

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in skip_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        group_name = "%s_%s" % (n, g_decay)

        if group_name not in param_group_names:
            txt_match = re.match(layerExtract, n)
            lr_decay_filter_text = 'text_encoder' in n and \
                                   (txt_match is None or
                                    (txt_match is not None and int(txt_match.group(1)) not in txt_fusion_layer))
            if 'visual_encoder' in n or lr_decay_filter_text:
                this_scale = lr_decay
            else:
                this_scale = 1.

            param_group_names[group_name] = {
                "weight_decay": this_decay,
                "lr_scale": this_scale,
                "params": [],
            }
            param_groups[group_name] = {
                "weight_decay": this_decay,
                "lr_scale": this_scale,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())


def add_lr_decay(model, name, lr_decay=0.75):
    """add lr decay"""
    param_group_names = {}
    param_groups = {}

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        group_name = n

        if group_name not in param_group_names:
            if name == 'encoder' or 'encoder' in group_name:
                this_scale = lr_decay
            else:
                this_scale = 1

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())


def create_optimizer(args, models, model_type, filter_bias_and_bn=True, is_distributed=False):
    opt_lower = args.opt.lower()
    if 'model' in model_type:
        # train whole model
        lr = args.lr
        opt_lower = args.opt.lower()
    else:
        print('error: no such model type')
        exit(1)

    parameters = []
    for model, name in zip(models, model_type):
        weight_decay = args.weight_decay
        if name != 'alpha':
            lr_decay = args.lr_decay
            if weight_decay and filter_bias_and_bn:
                skip = {}
                if hasattr(model, 'no_weight_decay'):
                    skip = model.no_weight_decay()
                parameters.extend(add_decay(model, name, weight_decay,
                                            lr_decay=lr_decay, skip_list=skip, is_distributed=is_distributed))
                weight_decay = 0.
            else:
                parameters.extend(add_lr_decay(model, name, lr_decay))
        else:
            parameters.append(model)

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas
    if hasattr(args, 'opt_args') and args.opt_args is not None:
        opt_args.update(args.opt_args)

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':        
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':        
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'novograd':
        optimizer = NovoGrad(parameters, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer
