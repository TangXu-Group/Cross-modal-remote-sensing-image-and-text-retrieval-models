from vilt.datasets import SYDNEYCaptionKarpathyDataset
from .datamodule_base import BaseDataModule


class SydneyCaptionKarpathyDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return SYDNEYCaptionKarpathyDataset

    @property
    def dataset_cls_no_false(self):
        return SYDNEYCaptionKarpathyDataset

    @property
    def dataset_name(self):
        return "sydney"
