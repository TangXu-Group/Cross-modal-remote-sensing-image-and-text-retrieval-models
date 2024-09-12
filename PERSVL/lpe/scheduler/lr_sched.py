import math


def adjust_learning_rate(optimizer, epoch, args, name):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if name == 'model':
        tlr = args.lr
    else:
        print('error: no such module')
        exit(1)

    if epoch < args.warmup_epochs:
        lr = tlr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (tlr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr