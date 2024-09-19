# PERSVL

Code for the TGRS 2024 paper: "Prior-Experience-based Vision-Language Model for Remote Sensing Image-Text Retrieval"

## Requirement
```bash
python== 
pytorch==
```

## Train
### Stage 1
See `run_{dataset name}_irtr_exp_CRSRT_backbone_only.sh` in `scripts` directory.

### Stage 2
See `run_{dataset name}_irtr_exp_CRSRT_LPE.sh` in `scripts` directory.

## Test
Please set `--evaluate=True --resume=<ckpt_path>`

## Contact for Issues
If you have any questions, you can send me an email. My mail address is 22171214766@stu.xidian.edu.cn.

## Acknowledge
Parts of this code were based on the codebase of [`ALBEF`](https://github.com/salesforce/ALBEF) and [`Swin-Transformer`](https://github.com/microsoft/Swin-Transformer), we gratefully thank the authors for their wonderful works.
