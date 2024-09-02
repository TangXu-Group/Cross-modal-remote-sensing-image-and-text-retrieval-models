import random
import torch
import io
import pyarrow as pa
import os
import numpy
from PIL import Image
from vilt.transforms import keys_to_transforms

import model.clip as clip
class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        transform_keys: str,
        image_size: int,
        names: list,
        text_column_name: str = "",
        remove_duplicate=True,
        max_text_len=77,
        prompt_length=16,
        draw_false_image=0,
        draw_false_text=0,
        image_only=False,
    ):
        """
        data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
        transform_keys : keys for generating augmented views of images
        text_column_name : pyarrow table column name that has list of strings as elements
        """
        assert len(transform_keys) >= 1
        super().__init__()

        self.transforms = transform_keys
        self.text_column_name = text_column_name
        self.names = names

        self.max_text_len = max_text_len
        self.prompt_length = prompt_length
        self.text_len = int(max_text_len-prompt_length)

        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        self.image_only = image_only
        self.data_dir = data_dir

        # print("this is text_column_name:{}".format(text_column_name))
        if len(names) != 0:
            tables = [
                pa.ipc.RecordBatchFileReader(
                    pa.memory_map(f"{data_dir}/{name}.arrow", "r")
                ).read_all()
                for name in names
                if os.path.isfile(f"{data_dir}/{name}.arrow")
            ]

            self.table_names = list()
            for i, name in enumerate(names):
                self.table_names += [name] * len(tables[i])
            # print("this is the self.table_names:{}".format(self.table_names))
            self.table = pa.concat_tables(tables, promote=True)
            if text_column_name != "":
                self.text_column_name = text_column_name
                self.all_texts = self.table[text_column_name].to_pandas().tolist()
                self.all_texts = (
                    [list(set(texts)) for texts in self.all_texts]
                    if remove_duplicate
                    else self.all_texts
                )
            else:
                self.all_texts = list()

        else:
            self.all_texts = list()


        self.index_mapper = dict()



        if text_column_name != "" and not self.image_only:
            j = 0
            for i, texts in enumerate(self.all_texts):

                for _j in range(len(texts)):
                    self.index_mapper[j] = (i, _j)
                    j += 1

        else:
            for i in range(len(self.table)):
                self.index_mapper[i] = (i, None)

        model_name = str(self.transforms)
        _, self.preprocess = clip.load(model_name, device='cpu')

        self.single_text = []
        for i in range(len(self.all_texts)):
            for j in range(len(self.all_texts[i])):
                self.single_text.append(self.all_texts[i][j])
        self.single_text = set(self.single_text)

        self.single_text_plug = list()

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.index_mapper)

    def get_raw_image(self, index, image_key="image"):
        index, caption_index = self.index_mapper[index]
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)
        return Image.open(image_bytes).convert("RGB")




    def get_image(self, index, image_key="image"):
        # print("this is index:{}".format(index))
        image = self.get_raw_image(index, image_key=image_key)
        clip_img = self.preprocess(image)
        _index, caption_index = self.index_mapper[index]
        image_path = self.table["path"][_index]


        return {
            "clip_img": clip_img,
            "img_index": self.index_mapper[index][0],
            "cap_index": self.index_mapper[index][1],
            "raw_index": index,
            "path": image_path
        }

    def get_false_image(self, rep, image_key="image"):
        random_index = random.randint(0, len(self.index_mapper) - 1)
        image = self.get_raw_image(random_index, image_key=image_key)
        image_tensor = [tr(image) for tr in self.transforms]
        return {f"false_image_{rep}": image_tensor}

    def get_text(self, raw_index):
        index, caption_index = self.index_mapper[raw_index]
        text = self.all_texts[index][caption_index]
        clip_text = clip.tokenize(text, context_length=77-self.prompt_length)

        return {
            "clip_text": (text, clip_text),
            "img_index": index,
            "cap_index": caption_index,
            "raw_index": raw_index,
        }

    def get_false_text(self, rep):
        random_index = random.randint(0, len(self.index_mapper) - 1)

        index, caption_index = self.index_mapper[random_index]
        text = self.all_texts[index][caption_index]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {f"false_text_{rep}": (text, encoding)}

    def get_single_text(self):
        for i, item in enumerate(self.single_text):
            item_encoder = clip.tokenize(item, context_length=77-self.prompt_length)
            self.single_text_plug.append(item_encoder[0].tolist())

        return self.single_text_plug

    def get_suite(self, index):
        result = None
        while result is None:

            ret = dict()

            ret.update(self.get_image(index))

            if not self.image_only:
                txt = self.get_text(index)
                ret.update({"replica": True if txt["cap_index"] > 0 else False})
                ret.update(txt)

            result = True

        return ret


    def collate(self, batch, mlm_collator):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        img_keys = [k for k in list(dict_batch.keys()) if "clip_img" in k]

        for img_key in img_keys:
            images = dict_batch[img_key]
            img = [img_i.cpu().numpy() for img_i in images]
            img = numpy.array(img)
            img = torch.tensor(img)
            dict_batch[img_key] = img

        txt_keys = [k for k in list(dict_batch.keys()) if "clip_text" in k]
        for text_key in txt_keys:
            txt = dict_batch[text_key]
            text = [_txt for (_txt,_clip_txt) in txt]
            clip_txt = torch.cat(
                [_clip_txt
                 for (_txt, _clip_txt) in txt],
                dim=0
            )
            dict_batch[f"{text_key}_txt"] = text
            dict_batch[f"{text_key}_token"] = clip_txt

        return dict_batch
