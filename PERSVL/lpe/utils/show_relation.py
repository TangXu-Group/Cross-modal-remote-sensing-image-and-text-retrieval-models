import os

import torch
import math
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def show_relation_func(args, model, data_loader, tokenizer, device, text_splitor, cur_output_dir):
    for image, text, image_path in data_loader:

        print('img path: ', image_path)
        print('text: ', text)

        image = image.to(device, non_blocking=True)
        # idx = idx.to(device, non_blocking=True)
        text_input = tokenizer(text, padding='longest', max_length=30, return_tensors="pt").to(device)

        # if args.use_alter_mlm:
        #     noun_list = []
        #     noun_idxs = []
        #     for txt in text:
        #         text_doc = text_splitor(txt)
        #         list_idxs = []
        #         list_noun = []
        #         for token in text_doc:
        #             if token.is_stop is not True or token.pos_ == 'NUM':
        #                 list_idxs.append(token.i + 1)       # the input of bert contain a head 'cls'
        #                 list_noun.append(token.text)
        #         noun_idxs.append(list_idxs)
        #         noun_list.append(list_noun)
        # else:
        #     noun_idxs = None

        output = model.direct_forward(image, text_input)

        # image_embeds, _, low_level_feat, high_level_feat = model.get_img_embeds(image, detail=True)

        # print('=============================================')

        print('============================================')
        # show_img(image, img_path)
        # show_picture(image_embeds[:, 1:, :], 'fusion')
        # show_picture(low_level_feat, 'low')
        # show_picture(high_level_feat, 'high')
        # _, _, _, _, _, _, _, _ = model(image, text_input, idx=idx, noun_idxs=noun_idxs)


def show_img(img, img_path):
    transform_img = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    ])
    # img = img.squeeze(0)
    # img = img.permute(1, 2, 0)
    # img = img.cpu().detach().numpy()
    #
    # tmp = img[:, :, 2]
    # img[:, :, 2] = img[:, :, 1]
    # img[:, :, 1] = img[:, :, 0]
    # img[:, :, 0] = tmp

    image = Image.open(img_path[0]).convert('RGB')
    img = transform_img(image)


    plt.figure()
    plt.title(img_path)
    plt.imshow(img)
    plt.show()


def show_picture(feat, name):
    feat = feat.squeeze(0)
    width = int(math.sqrt(feat.shape[0]))
    feat = feat.mean(1).reshape(width, width)
    feat = feat.cpu().detach().numpy()

    plt.figure()
    plt.title(name)
    plt.imshow(feat)
    plt.show()



