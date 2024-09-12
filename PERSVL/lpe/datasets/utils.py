import re
import torch
import numpy as np
import torchvision.transforms as T
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask


class SimMIMTransform:
    def __init__(self, config, mode):
        self.mode = mode
        if mode == 'train':
            self.transform_img = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD)),
            ])

            model_patch_size = config.MODEL.SWIN.PATCH_SIZE

            self.mask_generator = MaskGenerator(
                input_size=config.DATA.IMG_SIZE,
                mask_patch_size=config.DATA.MASK_PATCH_SIZE,
                model_patch_size=model_patch_size,
                mask_ratio=config.DATA.MASK_RATIO,
            )
        else:
            self.transform_img = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                         interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD)),
            ])

    def __call__(self, img):
        img = self.transform_img(img)
        if self.mode == 'train':
            mask = self.mask_generator()
            return img, mask
        else:
            return img


def pre_caption(caption, max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption
        
        
        