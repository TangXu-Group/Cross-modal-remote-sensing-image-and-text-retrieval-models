import json
import os
from PIL import Image
from torch.utils.data import Dataset
from .utils import pre_caption
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.img_ids = {}
        self.txt2id_dict = {}
        self.id2txt_dict = {}
        self.text = []
        self.txt2img = {}
        self.img2txt = {}

        n = 0
        txt_id = -1
        for ann in self.ann:
            img_id = ann['image_id']
            caption = pre_caption(ann['caption'], self.max_words)
            cur_txt_id = self.txt2id_dict.get(caption, None)

            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

            if cur_txt_id is None:
                txt_id += 1
                cur_txt_id = txt_id
                self.txt2id_dict[caption] = cur_txt_id
                self.id2txt_dict[cur_txt_id] = []
                self.txt2img[cur_txt_id] = {}
            self.text.append(caption)
            self.id2txt_dict[cur_txt_id].append(len(self.text) - 1)

            if self.img2txt.get(self.img_ids[img_id], None) is None:
                self.img2txt[self.img_ids[img_id]] = []
            self.img2txt[self.img_ids[img_id]].append(cur_txt_id)
            self.txt2img[cur_txt_id][self.img_ids[img_id]] = True

        # for k, v in self.txt2img.items():
        #     self.txt2img[k] = list(set(v))

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image, mask = self.transform(image)

        caption = pre_caption(ann['caption'], self.max_words)

        return image, caption, mask, self.img_ids[ann['image_id']]


class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.txt2id_dict = {}
        self.id2txt_dict = {}
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = -1
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                cur_text = pre_caption(caption, self.max_words)
                cur_txt_id = self.txt2id_dict.get(cur_text, None)
                if cur_txt_id is None:
                    txt_id += 1
                    cur_txt_id = txt_id
                    self.txt2id_dict[cur_text] = cur_txt_id
                    self.txt2img[cur_txt_id] = []
                    self.id2txt_dict[cur_txt_id] = []
                self.text.append(cur_text)
                self.id2txt_dict[cur_txt_id].append(len(self.text) - 1)
                self.img2txt[img_id].append(cur_txt_id)
                self.txt2img[cur_txt_id].append(img_id)


        # self.text = []
        # self.image = []
        # self.txt2img = {}
        # self.img2txt = {}
        #
        # txt_id = 0
        # for img_id, ann in enumerate(self.ann):
        #     self.image.append(ann['image'])
        #     self.img2txt[img_id] = []
        #     for i, caption in enumerate(ann['caption']):
        #         self.text.append(pre_caption(caption, self.max_words))
        #         self.img2txt[img_id].append(txt_id)
        #         self.txt2img[txt_id] = img_id
        #         txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index


class re_show_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, 'r'))
        self.ann_new = []
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}

        for tann in self.ann:
            for caption in tann['caption']:
                tmp_dict = {}
                tmp_dict['image'] = tann['image']
                tmp_dict['caption'] = caption
                self.ann_new.append(tmp_dict)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann_new[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = pre_caption(ann['caption'], self.max_words)

        return image, caption, image_path


if __name__ == '__main__':
    # args = get_args_parser()
    #
    # json_root = 'json_root'
    # train_file = [os.path.join(args.data_dir, os.path.join(json_root, 'rsitmd_train.json'))]
    # val_file = os.path.join(args.data_dir, os.path.join(json_root, 'rsitmd_test.json'))
    # test_file = os.path.join(args.data_dir, os.path.join(json_root, 'rsitmd_test.json'))
    #
    # image_root = os.path.join(args.data_dir, 'root/images')
    #
    # train_dataset = re_train_dataset(train_file, clip_transform, image_root)
    # val_dataset = re_eval_dataset(val_file, clip_transform, image_root)
    # test_dataset = re_eval_dataset(test_file, clip_transform, image_root)

    print(1)
