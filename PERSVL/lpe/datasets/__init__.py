import os
from .caption_dataset import re_train_dataset, re_eval_dataset, re_show_dataset
from torch.utils.data import DataLoader
import torch

from .utils import SimMIMTransform


def create_datasets(args, swin_cfg):
    train_transform = SimMIMTransform(swin_cfg, 'train')
    test_transform = SimMIMTransform(swin_cfg, 'test')

    for i in range(len(args.train_file)):
        args.train_file[i] = os.path.join(args.data_dir, args.train_file[i])
    args.val_file = os.path.join(args.data_dir, args.val_file)
    args.test_file = os.path.join(args.data_dir, args.test_file)

    train_dataset = re_train_dataset(
        ann_file=args.train_file,
        transform=train_transform,
        image_root=args.image_dir,
    )
    val_dataset = re_eval_dataset(
        ann_file=args.val_file,
        transform=test_transform,
        image_root=args.image_dir,
    )
    test_dataset = re_eval_dataset(
        ann_file=args.test_file,
        transform=test_transform,
        image_root=args.image_dir,
    )
    return train_dataset, val_dataset, test_dataset


def create_show_datasets(args, swin_cfg):
    test_transform = SimMIMTransform(swin_cfg, 'test')

    args.test_file = os.path.join(args.data_dir, args.test_file)

    dataset = re_show_dataset(
        ann_file=args.test_file,
        transform=test_transform,
        image_root=args.image_dir,
    )

    return dataset


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=False,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders