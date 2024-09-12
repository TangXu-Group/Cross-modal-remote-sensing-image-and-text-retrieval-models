import json
import os
# import pandas as pd
# import pyarrow as pa
import random

from tqdm import tqdm
from glob import glob
from collections import defaultdict


def path2rest(path, iid2captions, iid2split, iid2imgid) -> list:
    """

    Args:
        path: image path
        iid2captions: image caption
        iid2split: image split (train, test, val)
        iid2imgid: image id

    Returns: list(name, captions, imgid, split)

    """
    name = path.split("/")[-1]
    captions = iid2captions[name]
    split = iid2split[name]
    imgid = iid2imgid[name]
    return [name, captions, imgid, split]


def process_train(batches):
    result_list = []
    for batch in batches:
        for i in range(len(batch[1])):
            tmp_dict = {}
            tmp_dict['caption'] = batch[1][i]
            tmp_dict['image'] = batch[0]
            tmp_dict['image_id'] = batch[2]
            result_list.append(tmp_dict)
    return result_list


def process_eval(batches):
    result_list = []
    for batch in batches:
        tmp_dict = {}
        tmp_dict['image'] = batch[0]
        tmp_dict['caption'] = batch[1]
        result_list.append(tmp_dict)
    return result_list


def make_arrow(root, dataset_root):
    with open(f"{root}/dataset.json", "r") as fp:
        captions = json.load(fp)

    captions = captions["images"]

    iid2captions = defaultdict(list)
    iid2split = dict()
    iid2imgid = dict()

    for cap in tqdm(captions):
        filename = cap["filename"]
        iid2split[filename] = cap["split"]
        iid2imgid[filename] = "sydney_" + str(cap["imgid"])
        for c in cap["sentences"]:
            iid2captions[filename].append(c["raw"])

    paths = list(glob(f"{root}/imgs/*.tif"))
    random.shuffle(paths)
    caption_paths = [path for path in paths if path.split("/")[-1] in iid2captions]

    if len(paths) == len(caption_paths):
        print("all images have caption annotations")
    else:
        print("not all images have caption annotations")
    print(
        len(paths), len(caption_paths), len(iid2captions),
    )

    bs = [path2rest(path, iid2captions, iid2split, iid2imgid) for path in tqdm(caption_paths)]

    for split in ["train", "val", "test"]:
        batches = [b for b in bs if b[-1] == split]

        filename = "sydney_" + split + ".json"
        filepath = os.path.join(dataset_root, filename)

        if split == "train":
            final_batch = process_train(batches)
        else:
            final_batch = process_eval(batches)

        os.makedirs(dataset_root, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(final_batch, f)

        # dataframe = pd.DataFrame(
        #     batches, columns=["image", "caption", "image_id", "split"],
        # )
        #
        #
        # table = pa.Table.from_pandas(dataframe)
        # os.makedirs(dataset_root, exist_ok=True)
        # with pa.OSFile(
        #     f"{dataset_root}/rsitmd_caption_karpathy_{split}.json", "wb"
        # ) as sink:
        #     with pa.RecordBatchFileWriter(sink, table.schema) as writer:
        #         writer.write_table(table)


if __name__ == '__main__':
    dir = r'/data1/amax/datasets/Sydney_captions/'
    make_arrow(dir + 'root', dir + 'json_root')