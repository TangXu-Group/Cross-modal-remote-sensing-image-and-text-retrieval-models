import torch
import torch.nn as nn
import pytorch_lightning as pl
from vilt.modules import heads, objectives, vilt_utils
# import model.prompt_clip as prompt_clip
import numpy as np
from PIL import Image
from model.context_cluster import cluster_model
from torch_ema import ExponentialMovingAverage
from prompt_clip import clip
from collections import OrderedDict
import math
from torch.nn import Conv2d, Dropout
from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x)
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        # print("this is the shape of x:{}".format(x.shape))
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.test_only = False
        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        print("\n###########This is config ####################")
        print(self.hparams.config)
        print("###########This is config ####################\n")

        self.loss_scale = self.hparams.config["loss_scale"]

        self.name = config["clip_model"]
        self.model, self.preprocess = clip.load(self.name)
        self.model = self.model.float()

        self.prompt_length = config['prompt_length']
        self.prompt_dropout = Dropout(0.1)

        self.loss_scale_kl = self.hparams.config["loss_scale_kl"]
        self.loss_scale_un = self.hparams.config["loss_scale_un"]



        """
        for text prompt
        """

        prompt_dim = config["txt_features_dim"]
        self.prompt_proj = nn.Linear(
            prompt_dim, prompt_dim)
        nn.init.kaiming_normal_(
            self.prompt_proj.weight, a=0, mode='fan_out')

        val = math.sqrt(6. / float(1 + prompt_dim))  # noqa

        self.prompt_embeddings = nn.Parameter(torch.zeros(
            1, self.prompt_length, prompt_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

        """
        for visual_ prompt
        """

        self.visual_prompt_dropout = Dropout(0.1)
        visual_patch_size = config['image_size']

        visual_prompt_dim = config['img_features_dim']
        self.visual_prompt_proj = nn.Linear(
            visual_prompt_dim, visual_prompt_dim)
        nn.init.kaiming_normal_(
            self.visual_prompt_proj.weight, a=0, mode='fan_out')

        visual_val = math.sqrt(6. / float(3 * reduce(mul, visual_patch_size, 1) + visual_prompt_dim))  # noqa

        self.visual_prompt_embeddings = nn.Parameter(torch.zeros(
            1, self.prompt_length, visual_prompt_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.visual_prompt_embeddings.data, -visual_val, visual_val)

        self.visual_prompt_gather_dropout = Dropout(0.1)
        self.visual_prompt_proj_gather = nn.Linear(
            visual_prompt_dim, visual_prompt_dim)
        nn.init.kaiming_normal_(
            self.visual_prompt_proj_gather.weight, a=0, mode='fan_out')

        # xavier_uniform initialization
        nn.init.uniform_(self.visual_prompt_embeddings.data, -visual_val, visual_val)


        self.analysis_img = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim),
        )
        self.analysis_txt = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim),
        )
        self.prior_img = nn.Linear(visual_prompt_dim, prompt_dim)

        self.prior_analysis_img = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim),
        )
        self.cluster_model = cluster_model(num_classes=visual_prompt_dim)
        self.cluster_mapping = nn.Linear(visual_prompt_dim, 1)

        self.adapter_img = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim),
            nn.GELU(),
            nn.Linear(prompt_dim, prompt_dim),
        )
        self.adapter_img_mapping = nn.Sequential(
            nn.Linear(prompt_dim, 1),
        )

        self.adapter_txt = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim),
            nn.GELU(),
            nn.Linear(prompt_dim, prompt_dim),
        )
        self.adapter_txt_mapping = nn.Sequential(
            nn.Linear(prompt_dim, 1),
        )


        # ===================== Test_only ======================
        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)


    def infer(
            self,
            batch,
    ):

        (img, txt) = batch
        B = txt.shape[0]
        visual_prior_prompt = self.cluster_model(img)
        visual_prior_prompt_out = self.prior_img(visual_prior_prompt)
        visual_prior_prompt_prior = torch.clamp(self.prior_analysis_img(visual_prior_prompt_out), min=0)

        visual_prior_prompt = visual_prior_prompt.unsqueeze(dim=1).repeat(1, self.prompt_length, 1)
        visual_ratio = torch.clip(self.cluster_mapping(visual_prior_prompt), min=0, max=1)
        visual_random_prompt = self.visual_prompt_dropout(
            self.visual_prompt_proj(self.visual_prompt_embeddings).expand(B, -1, -1))
        visual_prompt = (1 - visual_ratio) * visual_prior_prompt +  visual_ratio *visual_random_prompt
        visual_prompt = self.visual_prompt_gather_dropout(self.visual_prompt_proj_gather(visual_prompt))

        text_prompt = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1))

        logits_per_image, logits_per_text, image_features, text_features = self.model(img, txt, visual_prompt,
                                                                                      text_prompt)

        img_adapter = self.adapter_img(image_features)
        img_adapter_ratio = self.adapter_img_mapping(img_adapter)
        img_adapter_ratio = torch.clamp(img_adapter_ratio, min=0, max=0.1)
        image_features = img_adapter_ratio * img_adapter + (1 - img_adapter_ratio) * image_features

        txt_adapter = self.adapter_txt(text_features)
        txt_adapter_scale = self.adapter_txt_mapping(txt_adapter)
        txt_adapter_scale = torch.clamp(txt_adapter_scale, min=0, max=0.1)
        text_features = txt_adapter_scale * txt_adapter + (1 - txt_adapter_scale) * text_features

        img_analy = self.analysis_img(image_features)
        txt_analy = self.analysis_txt(text_features)

        img_analy = torch.clamp(img_analy, min=0)
        txt_analy = torch.clamp(txt_analy, min=0)

        loss_scale = self.loss_scale

        ret = {
            "visual_prior_prompt": visual_prior_prompt_out,
            "visual_prior_prompt_prior": visual_prior_prompt_prior,
            "loss_scale": loss_scale,
            "loss_scale_kl": self.loss_scale_kl,
            "loss_scale_un": self.loss_scale_un,
            "img_analysis": img_analy,
            "txt_analysis": txt_analy,
            "image_features": image_features,
            "text_features": text_features,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,

        }

        return ret

    def txt_embeds(
            self,
            batch
    ):
        txt = batch
        B = txt.shape[0]
        text_prompt = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1))
        text_features = self.model.encode_text(txt, text_prompt)

        txt_adapter = self.adapter_txt(text_features)
        txt_adapter_scale = self.adapter_txt_mapping(txt_adapter)
        txt_adapter_scale = torch.clamp(txt_adapter_scale, min=0, max=0.1)
        text_features = txt_adapter_scale * txt_adapter + (1 - txt_adapter_scale) * text_features

        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        ret = {
            "text_feats": text_features,

        }

        return ret

    def img_embeds(
            self,
            batch
    ):
        img = batch
        B = img.shape[0]

        visual_prior_prompt = self.cluster_model(img)
        visual_prior_prompt = visual_prior_prompt.unsqueeze(dim=1).repeat(1, self.prompt_length, 1)
        visual_ratio = torch.clamp(self.cluster_mapping(visual_prior_prompt), min=0, max=1)
        visual_random_prompt = self.visual_prompt_dropout(
            self.visual_prompt_proj(self.visual_prompt_embeddings).expand(B, -1, -1))
        visual_prompt = (1 - visual_ratio) * visual_prior_prompt + visual_ratio *visual_random_prompt
        visual_prompt = self.visual_prompt_gather_dropout(self.visual_prompt_proj_gather(visual_prompt))
        image_features = self.model.encode_image(img, visual_prompt)

        img_adapter = self.adapter_img(image_features)
        img_adapter_ratio = self.adapter_img_mapping(img_adapter)
        img_adapter_ratio = torch.clamp(img_adapter_ratio, min=0, max=0.1)
        image_features = img_adapter_ratio * img_adapter + (1 - img_adapter_ratio) * image_features

        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        ret = {
            "image_feats": image_features,

        }

        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "clip" in self.current_tasks:
            ret.update(objectives.compute_clip(self, batch))

        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_wpa(self, batch))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        # print(total_loss)
        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        # self.current_epoch
        # print("this is batch_ics:{}".format(self.current_epoch))
        self.hparams.config["current_epoch"] = self.current_epoch
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):

        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        return ret

    def test_epoch_end(self, outs):
        self.test_only = True

        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)
