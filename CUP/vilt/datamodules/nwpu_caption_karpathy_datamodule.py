from vilt.datasets import NWPUCaptionKarpathyDataset
from .datamodule_base import BaseDataModule


class NWPUCaptionKarpathyDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return NWPUCaptionKarpathyDataset

    @property
    def dataset_cls_no_false(self):
        return NWPUCaptionKarpathyDataset

    @property
    def dataset_name(self):
        return "nwpu"
