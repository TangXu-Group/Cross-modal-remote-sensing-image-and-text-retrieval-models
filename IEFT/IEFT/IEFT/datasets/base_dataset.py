import random
import torch
import io
import pyarrow as pa
import os

from PIL import Image
from IEFT.transforms import keys_to_transforms


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        transform_keys: list,
        image_size: int,
        names: list,
        text_column_name: str = "",
        remove_duplicate=True,
        max_text_len=40,
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
        # print(transform_keys)
        self.transforms = keys_to_transforms(transform_keys, size=image_size)
        self.text_column_name = text_column_name
        self.names = names
        self.max_text_len = max_text_len
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        self.image_only = image_only
        self.data_dir = data_dir

        if len(names) != 0:
            tables = [
                pa.ipc.RecordBatchFileReader(
                    pa.memory_map(f"{data_dir}/{name}.arrow", "r")
                ).read_all()
                for name in names
                if os.path.isfile(f"{data_dir}/{name}.arrow")
            ]
            # print(len(tables))
            # for name in names:
            #     print(f"{data_dir}/{name}.arrow")
            self.table_names = list()
            # print(names)
            # print("this is the length of names:{}".format(len(names)))
            for i, name in enumerate(names):
                # print(i)
                self.table_names += [name] * len(tables[i])
            # print("this is self.table_names:{}".format(self.table_names))
            self.table = pa.concat_tables(tables, promote=True)
            # print(self.table.columns)
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
        # print("This is self.index_mapper:{}".format(self.index_mapper))

        # print("this is keys:{}".format(self.table.keys()))

        self.single_text = []
        for i in range(len(self.all_texts)):
            for j in range(len(self.all_texts[i])):
                self.single_text.append(self.all_texts[i][j])

        self.single_text = set(self.single_text)
        self.single_text_plug = list()
        # print(len(self.table["path"]))
        # print(len(self.table["image"]))

    @property
    def corpus(self):
        # print("corpus")
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.index_mapper)

    def get_raw_image(self, index, image_key="image"):
        # print("get_raw_image")
        index, caption_index = self.index_mapper[index]
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)

        return Image.open(image_bytes).convert("RGB")

    def get_image(self, index, image_key="image"):
        # print("get_image")
        image = self.get_raw_image(index, image_key=image_key)
        image_tensor = [tr(image) for tr in self.transforms]
        _index, caption_index = self.index_mapper[index]
        image_path = self.table["path"][_index]
        return {
            "image": image_tensor,
            "img_index": self.index_mapper[index][0],
            "cap_index": self.index_mapper[index][1],
            "raw_index": index,
            "path": image_path
        }

    def get_false_image(self, rep, image_key="image"):
        # print("get_false_image")
        random_index = random.randint(0, len(self.index_mapper) - 1)
        image = self.get_raw_image(random_index, image_key=image_key)
        image_tensor = [tr(image) for tr in self.transforms]
        return {f"false_image_{rep}": image_tensor}

    def get_text(self, raw_index):
        # print("get text")
        index, caption_index = self.index_mapper[raw_index]

        text = self.all_texts[index][caption_index]
        # print("text:{}".format(text))
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        # print("encoder:{}".format(encoding))
        return {
            "text": (text, encoding),
            "img_index": index,
            "cap_index": caption_index,
            "raw_index": raw_index,
        }


    def get_single_text(self):
        for i, item in enumerate(self.single_text):
            item_encoder = self.tokenizer(
                item,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_special_tokens_mask=True,
            )
            self.single_text_plug.append(item_encoder["input_ids"])

        return self.single_text_plug

    #


    def get_false_text(self, rep):
        # print("get_false_text")
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

    def get_suite(self, index):
        # print("get_suite")
        result = None
        while result is None:
            ret = dict()
            ret.update(self.get_image(index))
            if not self.image_only:
                txt = self.get_text(index)
                # print("this is text:_{}".format(txt))
                ret.update({"replica": True if txt["cap_index"] > 0 else False})
                ret.update(txt)

            for i in range(self.draw_false_image):
                ret.update(self.get_false_image(i))
            for i in range(self.draw_false_text):
                ret.update(self.get_false_text(i))
            result = True

        return ret

    def collate(self, batch, mlm_collator):
        # print("collate")
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        # print(keys)
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        img_sizes = list()

        for img_key in img_keys:
            img = dict_batch[img_key]
            img_sizes += [ii.shape for i in img if i is not None for ii in i]

        for size in img_sizes:
            assert (
                len(size) == 3
            ), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"

        if len(img_keys) != 0:
            max_height = max([i[1] for i in img_sizes])
            max_width = max([i[2] for i in img_sizes])

        for img_key in img_keys:
            img = dict_batch[img_key]
            view_size = len(img[0])

            new_images = [
                torch.zeros(batch_size, 3, max_height, max_width)
                for _ in range(view_size)
            ]

            for bi in range(batch_size):
                orig_batch = img[bi]
                for vi in range(view_size):
                    if orig_batch is None:
                        new_images[vi][bi] = None
                    else:
                        orig = img[bi][vi]
                        new_images[vi][bi, :, : orig.shape[1], : orig.shape[2]] = orig

            dict_batch[img_key] = new_images

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

        if len(txt_keys) != 0:
            texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            draw_text_len = len(encodings)
            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = mlm_collator(flatten_encodings)

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],
                )

                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask

        return dict_batch
