from IEFT.datasets import RSICDCaptionKarpathyDataset
from .datamodule_base import BaseDataModule


class RSICDCaptionKarpathyDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return RSICDCaptionKarpathyDataset

    @property
    def dataset_cls_no_false(self):
        return RSICDCaptionKarpathyDataset

    @property
    def dataset_name(self):
        return "rsicd"
