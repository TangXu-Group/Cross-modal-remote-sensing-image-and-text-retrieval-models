from .base_dataset import BaseDataset


class RSICDCaptionKarpathyDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]

        if split == "train":
            names = ["rsicd_caption_karpathy_label_train"]
        elif split == "val":
            names = [ "rsicd_caption_karpathy_label_val"]
        elif split == "test":
            names = ["rsicd_caption_karpathy_label_test"]

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

    def __getitem__(self, index):

        return self.get_suite(index)
