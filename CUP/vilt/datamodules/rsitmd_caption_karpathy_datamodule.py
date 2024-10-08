from vilt.datasets import RSITMDCaptionKarpathyDataset
from .datamodule_base import BaseDataModule


class RSITMDCaptionKarpathyDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return RSITMDCaptionKarpathyDataset

    @property
    def dataset_cls_no_false(self):
        return RSITMDCaptionKarpathyDataset

    @property
    def dataset_name(self):
        return "rsitmd"
