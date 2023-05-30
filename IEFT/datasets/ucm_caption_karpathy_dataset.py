from .base_dataset import BaseDataset


class UCMCaptionKarpathyDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]

        if split == "train":
            names = ["ucm_caption_karpathy_new_train", "ucm_caption_karpathy_new_val"]
        elif split == "val":
            names = ["ucm_caption_karpathy_new_test"]
        elif split == "test":
            names = ["ucm_caption_karpathy_new_test"]

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

    def __getitem__(self, index):
        return self.get_suite(index)
