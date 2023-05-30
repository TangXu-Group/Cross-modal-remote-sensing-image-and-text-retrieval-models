from IEFT.datasets import UCMCaptionKarpathyDataset
from .datamodule_base import BaseDataModule


class UCMCaptionKarpathyDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return UCMCaptionKarpathyDataset

    @property
    def dataset_cls_no_false(self):
        return UCMCaptionKarpathyDataset

    @property
    def dataset_name(self):
        return "ucm"
