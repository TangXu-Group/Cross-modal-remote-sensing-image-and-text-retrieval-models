from .vg_caption_datamodule import VisualGenomeCaptionDataModule
from .f30k_caption_karpathy_datamodule import F30KCaptionKarpathyDataModule
from .coco_caption_karpathy_datamodule import CocoCaptionKarpathyDataModule
from .conceptual_caption_datamodule import ConceptualCaptionDataModule
from .sbu_datamodule import SBUCaptionDataModule
from .vqav2_datamodule import VQAv2DataModule
from .nlvr2_datamodule import NLVR2DataModule
from .rsicd_caption_karpathy_datamodule import RSICDCaptionKarpathyDataModule
from .sydney_caption_karpathy_datamodule import SydneyCaptionKarpathyDataModule
from .ucm_caption_karpathy_datamodule import UCMCaptionKarpathyDataModule
from .rsitmd_caption_karpathy_datamodule import RSITMDCaptionKarpathyDataModule
from .nwpu_caption_karpathy_datamodule import NWPUCaptionKarpathyDataModule

_datamodules = {
    "vg": VisualGenomeCaptionDataModule,
    "f30k": F30KCaptionKarpathyDataModule,
    "coco": CocoCaptionKarpathyDataModule,
    "gcc": ConceptualCaptionDataModule,
    "sbu": SBUCaptionDataModule,
    "vqa": VQAv2DataModule,
    "nlvr2": NLVR2DataModule,
    "rsicd":RSICDCaptionKarpathyDataModule,
    "sydney":SydneyCaptionKarpathyDataModule,
    "ucm":UCMCaptionKarpathyDataModule,
    "rsitmd":RSITMDCaptionKarpathyDataModule,
    "nwpu":NWPUCaptionKarpathyDataModule
}