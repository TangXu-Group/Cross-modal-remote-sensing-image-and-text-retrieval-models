import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools
import torch
from torch.utils.data.distributed import DistributedSampler
from einops import rearrange
import numpy as np
import PIL
from matplotlib import pyplot as plt
from vilt.modules.dist_utils import all_gather
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score, balanced_accuracy_score
import pandas as pd



def cost_matrix_cosine(x, y, eps=1e-5):
    """Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(b, n).sum(dim=-1, keepdim=False)
    return trace


@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2) / beta)

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T


def optimal_transport_dist(
    txt_emb, img_emb, txt_pad, img_pad, beta=0.5, iteration=50, k=1
):
    """ [B, M, D], [B, N, D], [B, M], [B, N]"""
    cost = cost_matrix_cosine(txt_emb, img_emb)
    # mask the padded inputs
    joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
    cost.masked_fill_(joint_pad, 0)

    txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)
    img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)

    T = ipot(
        cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad, beta, iteration, k
    )
    distance = trace(cost.matmul(T.detach()))
    return distance


def compute_mlm(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=True, mask_image=False)
    mlm_logits = pl_module.mlm_score(infer["text_feats"])
    mlm_labels = infer["text_labels"]

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(
        ret["mlm_logits"], ret["mlm_labels"]
    )
    pl_module.log(f"mlm/{phase}/loss", loss)
    pl_module.log(f"mlm/{phase}/accuracy", acc)

    return ret


def compute_mpp(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=True)
    mpp_logits = pl_module.mpp_score(infer["image_feats"])
    mpp_logits = torch.stack(
        [
            mpp_logits[:, :, 0:256],
            mpp_logits[:, :, 256:512],
            mpp_logits[:, :, 512:768],
        ],
        dim=2,
    )
    mpp_labels = infer["image_labels"]

    mpp_loss = F.cross_entropy(
        mpp_logits.view(-1, 256),
        mpp_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mpp_loss": mpp_loss,
        "mpp_logits": mpp_logits,
        "mpp_labels": mpp_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mpp_loss")(ret["mpp_loss"])
    acc = getattr(pl_module, f"{phase}_mpp_accuracy")(
        ret["mpp_logits"], ret["mpp_labels"]
    )
    pl_module.log(f"mpp/{phase}/loss", loss)
    pl_module.log(f"mpp/{phase}/accuracy", acc)

    return ret


def compute_mppd(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=True)
    mppd_logits = pl_module.mppd_score(infer["image_feats"])
    mppd_labels = infer["image_labels_mppd"]
    filter_to_train = infer["image_labels"].float().mean(dim=-1) != -100

    labels = mppd_labels[filter_to_train]
    logits = mppd_logits[filter_to_train]
    mppd_loss = F.mse_loss(logits, labels)

    ret = {
        "mppd_loss": mppd_loss,
        "mppd_logits": mppd_logits,
        "mppd_labels": mppd_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mppd_loss")(ret["mppd_loss"])
    pl_module.log(f"mppd/{phase}/loss", loss)

    return ret


def compute_mpfr(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=True)
    mpfr_logits = pl_module.mpfr_score(infer["image_feats"])
    mpfr_labels = infer["image_labels_mpfr"]
    filter_to_train = infer["image_labels"].float().mean(dim=-1) != -100

    labels = mpfr_labels[filter_to_train]
    logits = mpfr_logits[filter_to_train]
    mpfr_loss = F.mse_loss(logits, labels)

    ret = {
        "mpfr_loss": mpfr_loss,
        "mpfr_logits": mpfr_logits,
        "mpfr_labels": mpfr_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mpfr_loss")(ret["mpfr_loss"])
    pl_module.log(f"mpfr/{phase}/loss", loss)

    return ret


def compute_itm_wpa(pl_module, batch):
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )
    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

    itm_images = [
        torch.stack(
            [
                ti if itm_labels[i] == 1 else fi
                for i, (ti, fi) in enumerate(zip(bti, bfi))
            ]
        )
        for bti, bfi in zip(batch["image"], batch["false_image_0"])
    ]

    batch = {k: v for k, v in batch.items()}
    batch["image"] = itm_images

    infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    with torch.cuda.amp.autocast(enabled=False):
        txt_emb, img_emb = infer["text_feats"], infer["image_feats"]
        txt_mask, img_mask = infer["text_masks"].bool(), infer["image_masks"].bool()
        for i, _len in enumerate(txt_mask.sum(dim=1)):
            txt_mask[i, _len - 1] = False
        txt_mask[:, 0] = False
        img_mask[:, 0] = False
        if "deit" in pl_module.hparams.config["vit"]:
            img_mask[:, 1] = False
        txt_pad, img_pad = ~txt_mask, ~img_mask

        cost = cost_matrix_cosine(txt_emb.float(), img_emb.float())
        joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
        cost.masked_fill_(joint_pad, 0)

        txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(
            dtype=cost.dtype
        )
        img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(
            dtype=cost.dtype
        )
        T = ipot(
            cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad, 0.5, 50, 1
        )
        distance = trace(cost.matmul(T.detach()))

    dist_pos = distance.masked_select(itm_labels == 1)
    dist_neg = distance.masked_select(itm_labels == 0)
    ot_loss = (dist_pos.sum() - dist_neg.sum()) / (dist_pos.size(0) + dist_neg.size(0))

    itm_logits = pl_module.itm_score(infer["cls_feats"])
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        "itm_wpa_loss": 0.1 * ot_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itm_loss")(ret["itm_loss"])
    wpa_loss = getattr(pl_module, f"{phase}_itm_wpa_loss")(ret["itm_wpa_loss"])
    acc = getattr(pl_module, f"{phase}_itm_accuracy")(
        ret["itm_logits"], ret["itm_labels"]
    )
    pl_module.log(f"itm/{phase}/loss", loss)
    pl_module.log(f"itm/{phase}/wpa_loss", wpa_loss)
    pl_module.log(f"itm/{phase}/accuracy", acc)

    return ret


def compute_imgcls(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    imgcls_logits = pl_module.img_classifier(infer["cls_feats"])
    imgcls_labels = batch["label"]
    imgcls_labels = torch.tensor(imgcls_labels).to(pl_module.device).long()
    imgcls_loss = F.cross_entropy(imgcls_logits, imgcls_labels)

    ret = {
        "imgcls_loss": imgcls_loss,
        "imgcls_logits": imgcls_logits,
        "imgcls_labels": imgcls_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_imgcls_loss")(ret["imgcls_loss"])
    acc = getattr(pl_module, f"{phase}_imgcls_accuracy")(
        ret["imgcls_logits"], ret["imgcls_labels"]
    )
    pl_module.log(f"imgcls/{phase}/loss", loss)
    pl_module.log(f"imgcls/{phase}/accuracy", acc)

    return ret


def compute_vqa(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    vqa_logits = pl_module.vqa_classifier(infer["cls_feats"])
    vqa_targets = torch.zeros(
        len(vqa_logits), pl_module.hparams.config["vqav2_label_size"]
    ).to(pl_module.device)

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    vqa_loss = (
        F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets)
        * vqa_targets.shape[1]
    )  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{phase}_vqa_score")(
        ret["vqa_logits"], ret["vqa_targets"]
    )
    pl_module.log(f"vqa/{phase}/loss", loss)
    pl_module.log(f"vqa/{phase}/score", score)

    return ret


def compute_nlvr2(pl_module, batch):
    infer1 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=1
    )
    infer2 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=2
    )

    cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
    nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)

    nlvr2_labels = batch["answers"]
    nlvr2_labels = torch.tensor(nlvr2_labels).to(pl_module.device).long()
    nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels)

    ret = {
        "nlvr2_loss": nlvr2_loss,
        "nlvr2_logits": nlvr2_logits,
        "nlvr2_labels": nlvr2_labels,
    }

    phase = "train" if pl_module.training else "val"

    if phase == "train":
        loss = getattr(pl_module, f"{phase}_nlvr2_loss")(ret["nlvr2_loss"])
        acc = getattr(pl_module, f"{phase}_nlvr2_accuracy")(
            ret["nlvr2_logits"], ret["nlvr2_labels"]
        )
        pl_module.log(f"nlvr2/{phase}/loss", loss)
        pl_module.log(f"nlvr2/{phase}/accuracy", acc)
    else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            dev_loss = getattr(pl_module, f"dev_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
                )
            )
            dev_acc = getattr(pl_module, f"dev_nlvr2_accuracy")(
                ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
            )
            pl_module.log(f"nlvr2/dev/loss", dev_loss)
            pl_module.log(f"nlvr2/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_nlvr2_accuracy")(
                ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
            )
            pl_module.log(f"nlvr2/test/loss", test_loss)
            pl_module.log(f"nlvr2/test/accuracy", test_acc)

    return ret

def compute_clip(pl_module, batch):
    is_training_phase = pl_module.training
    # print("this is batch.keys:{}".format(batch.keys()))
    NUM_SAMPLES = 10
    imgs = batch["clip_img"]
    texts = batch["clip_text_token"]


    infer = pl_module.infer(
        (imgs,texts,)
        )

    loss_scale_kl = infer["loss_scale_kl"]
    loss_scale_un = infer["loss_scale_un"]
    image_features = infer["image_features"]
    text_features = infer["text_features"]
    #
    img_analy = infer["img_analysis"]
    txt_analy = infer["txt_analysis"]
    visual_prior_prompt_prior = infer["visual_prior_prompt_prior"]

    #
    visual_prior_prompt = infer["visual_prior_prompt"]
    logits_per_image = infer["logits_per_image"]
    logits_per_text = infer["logits_per_text"]
    #
    # # print("this is the shape of logit_analysis_img:{} // logit_analysis_txt:{} ".format(logit_analysis_img.shape, logit_analysis_txt.shape))
    total_img = torch.zeros((NUM_SAMPLES, logits_per_image.size(0), logits_per_image.size(1)))
    total_txt = torch.zeros((NUM_SAMPLES, logits_per_image.size(0), logits_per_image.size(1)))
    prob_total_img = torch.zeros((NUM_SAMPLES, logits_per_image.size(0), logits_per_image.size(1)))
    prob_total_txt = torch.zeros((NUM_SAMPLES, logits_per_image.size(0), logits_per_image.size(1)))
    prob_total_img_prompt = torch.zeros((NUM_SAMPLES, image_features.size(0), image_features.size(1)))
    prob_total_img_prior = torch.zeros((NUM_SAMPLES, image_features.size(0), image_features.size(1)))
    prob_toal_img_intra = torch.zeros((NUM_SAMPLES, image_features.size(0), image_features.size(1)))
    prob_toal_txt_intra = torch.zeros((NUM_SAMPLES, image_features.size(0), image_features.size(1)))

    for t in range(NUM_SAMPLES):

        img_epsilon = torch.randn(image_features.size()).cuda()
        img_logit = image_features + torch.mul(img_analy, img_epsilon)
        # img_logit = img_logit / img_logit.norm(dim=1, keepdim=True)

        img_epsilon_intra = torch.randn(image_features.size()).cuda()
        img_logit_intra = image_features + torch.mul(img_analy.detach(), img_epsilon_intra)

        img_epsilon_prompt = torch.randn(image_features.size()).cuda()
        img_logit_prompt = image_features + torch.mul(img_analy.detach(), img_epsilon_prompt)

        img_epsilon_prompt_prior = torch.randn(visual_prior_prompt.size()).cuda()
        img_logit_prior = visual_prior_prompt + torch.mul(visual_prior_prompt_prior, img_epsilon_prompt_prior)

        txt_epsilon = torch.randn(text_features.size()).cuda()
        txt_logit = text_features + torch.mul(txt_analy, txt_epsilon)
        # txt_logit = txt_logit / txt_logit.norm(dim=1, keepdim=True)

        txt_epsilon_intra = torch.randn(text_features.size()).cuda()
        txt_logit_intra = text_features + torch.mul(txt_analy.detach(), txt_epsilon_intra)

        logit_scale = pl_module.model.logit_scale.exp()
        logit_img = logit_scale * img_logit @ txt_logit.t()
        logit_txt = logit_img.t()

        img_intra = img_logit_intra
        txt_intra = txt_logit_intra

        total_img = F.softmax(img_logit, dim=1)
        total_txt = F.softmax(txt_logit, dim=1)

        prob_total_img[t] = F.softmax(logit_img, dim=1)
        prob_total_txt[t] = F.softmax(logit_txt, dim=1)

        prob_toal_img_intra[t] = F.softmax(img_intra, dim=1)
        prob_toal_txt_intra[t] = F.softmax(txt_intra, dim=1)

        prob_total_img_prompt[t] = F.softmax(img_logit_prompt, dim=1)
        prob_total_img_prior[t] = F.softmax(img_logit_prior, dim=1)

    total_img_ave = torch.mean(total_img, 0).cuda()
    total_txt_ave = torch.mean(total_txt, 0).cuda()

    prob_total_img_ave = torch.mean(prob_total_img, 0).cuda()
    prob_total_txt_ave = torch.mean(prob_total_txt, 0).cuda()
    prob_total_img_prompt = torch.mean(prob_total_img_prompt, 0).cuda()
    prob_total_img_prior = torch.mean(prob_total_img_prior, 0).cuda()
    prob_toal_img_intra_ave = torch.mean(prob_toal_img_intra, 0).cuda()
    prob_toal_txt_intra_ave = torch.mean(prob_toal_txt_intra, 0).cuda()
    _b, _ = logits_per_image.shape
    label1 = torch.arange(_b).cuda()

    criterion3 = nn.NLLLoss().cuda()
    criterion4 = nn.NLLLoss().cuda()


    loss_img = criterion3(torch.log(prob_total_img_ave), label1)
    loss_text = criterion4(torch.log(prob_total_txt_ave), label1)
    clip_loss = (loss_img + loss_text)/2

    prob_toal_img_intra_ave = prob_toal_img_intra_ave @ prob_toal_img_intra_ave.t()
    prob_toal_txt_intra_ave = prob_toal_txt_intra_ave @ prob_toal_txt_intra_ave.t()

    uncertainty_loss = torch.abs(prob_toal_img_intra_ave - prob_toal_txt_intra_ave).mean()

    prob_total_img_prompt = prob_total_img_prompt.detach()

    prior_kl = F.kl_div(torch.log(prob_total_img_prior), prob_total_img_prompt)

    ret = {
        "clip_loss": clip_loss,
        "prior_loss":  loss_scale_kl * prior_kl,
        "uncertainty_loss": loss_scale_un * uncertainty_loss,
    }

    phase = "train" if pl_module.training else "val"

    clip_loss = getattr(pl_module, f"{phase}_clip_loss")(ret["clip_loss"])
    prior_loss = getattr(pl_module, f"{phase}_prior_loss")(ret["prior_loss"])
    uncertainty_loss = getattr(pl_module, f"{phase}_uncertainty_loss")(ret["uncertainty_loss"])

    pl_module.log(f"clip/{phase}/clip_loss", clip_loss)
    pl_module.log(f"clip/{phase}/prior_loss", prior_loss)
    pl_module.log(f"clip/{phase}/uncertainty_loss", uncertainty_loss)

    return ret

def svd(X, n_components=2):

    U, S, Vt = np.linalg.svd(X)

    return U[:, :n_components] * S[:n_components]




@torch.no_grad()
def compute_clip_recall(pl_module):
    # print(pl_module.hparams.config["test_only"])
    if pl_module.test_only:
        text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_test_dset()
    else:
        text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    # text_class = text_dset.id_to_class
    # all_class = text_dset.class_to_id

    _batch_size = len(text_dset)

    cls_number = pl_module.hparams.config["cls_number"]
    # text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=64,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=None,
        ),
    )

    text_preload = list()
    need_text_all = list()
    # text_class_id = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        # print("this is the keys for text_preload:{}".format(len(_b)))
        text_preload.append(
            {
                "text": _b["clip_text"],
                "clip_text": _b["clip_text_token"].to(pl_module.device),
                "img_index": _b["img_index"],
                "clip_img": _b["clip_img"].to(pl_module.device),
                "cap_index": _b["cap_index"],
                "raw_index": _b["raw_index"],
                # "txt_label": _b["txt_label"],
            }
        )

        for t_ in range(len(_b["clip_text"])):
            need_text_all.append([_b["clip_text"][t_]])
            # txt_label_word = text_class[_b["txt_label"][t_]]
            # text_class_id.append(txt_label_word )

    text_repeat_correspondence = {}
    rep_texi_id = []
    rep_img_id = []
    text_num=0
    all_repeat_text = text_dset.get_single_text()
    for batch_text_ids in tqdm.tqdm(text_preload, desc="building text correspondence"):
        for i in range(len(batch_text_ids["clip_text"])):
            text_ids = batch_text_ids["clip_text"][i]
            image_index_ = batch_text_ids["img_index"][i]

            text_ids = text_ids.tolist()
            index= all_repeat_text.index(text_ids)

            text_repeat_correspondence[text_num] = (image_index_, index)
            rep_texi_id.append(index)
            rep_img_id.append(image_index_)
            text_num += 1


    image_correspondence = {}
    text_values = [text_idx for k, (img_idx, text_idx) in text_repeat_correspondence.items()]
    for i in range(len(text_values)):
        img_rep_index = [
            img_idx
            for k, (img_idx, text_idx) in text_repeat_correspondence.items()
            if text_idx==text_values[i]
        ]
        # print(img_rep_index)
        image_correspondence[text_values[i]] = img_rep_index
    image_correspondence = sorted(image_correspondence.items(), key=lambda d: d[0])


    text_correspondence = {}
    key_values = [text_idx for k, (img_idx, text_idx) in text_repeat_correspondence.items()]
    for i in range(len(key_values)):
        text_rep_index = [
            img_idx
            for k, (img_idx, text_idx) in text_repeat_correspondence.items()
            if text_idx==key_values[i]
        ]
        # print(img_rep_index))
        text_correspondence["{}".format(key_values[i])] = text_rep_index


    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)
    dimension_number = pl_module.hparams.config["txt_features_dim"]
    txt_feats = torch.zeros(size=[len(text_dset), dimension_number])
    img_feats_visual = torch.zeros(size=[len(text_dset), dimension_number])

    # txt_feats_label = torch.zeros(size=[len(text_dset)])
    txt_feats_iids = torch.zeros(size=[len(text_dset)])
    txt_feats_cls = torch.zeros(size=[len(text_dset), cls_number])
    for i, _batch_txt in tqdm.tqdm(enumerate(text_preload), desc="text rank loop"):
        _t = _batch_txt["clip_text"]
        _i = _batch_txt["clip_img"]
        _iid =  _batch_txt["img_index"]
        _iid = torch.tensor(_iid)
        # _txt_label = torch.tensor(_batch_txt["txt_label"])
        out = pl_module.txt_embeds(_t)
        txt = out["text_feats"]
        # txt_cls = out["cls_txt"]

        img_ = pl_module.img_embeds(_i)["image_feats"]
        _l, _ = txt.shape


        txt_feats[i*64:i*64+_l] = txt
        # txt_feats_label[i*64:i*64+_l] = _txt_label
        img_feats_visual[i*64:i*64+_l] = img_
        txt_feats_iids[i*64:i*64+_l] = _iid

    if pl_module.test_only:
        image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_test_dset(
            image_only=True
        )
    else:
        image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
            image_only=True
        )

    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    # dist_sampler = None
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=64,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=None,
        ),
    )
    # img_class = image_dset.id_to_class

    img_feats = torch.zeros(size=[len(image_dset), dimension_number])
    img_feats_cls = torch.zeros(size=[len(image_dset), cls_number])
    # img_feats_label = torch.zeros(size=[len(image_dset)])

    rank_iids = torch.zeros(size=[len(image_dset)])
    img_preload = list()
    img_path_id = list()
    # img_class_id = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):

        for j in range(len(_b["path"])):
            path_id = (_b["path"][j], _b["img_index"][j])
            # print("this is path_id:{}".format(path_id))
            img_path_id.append([path_id])
            # img_class_id.append(img_class[_b["img_label"][j]])

        img_preload.append(
            {
                "img_index": _b["img_index"],
                "clip_img": _b["clip_img"].to(pl_module.device),
                "cap_index": _b["cap_index"],
                "raw_index": _b["raw_index"],
                # "img_label": _b["img_label"],
            }
        )


    for i, _batch_img in tqdm.tqdm(enumerate(img_preload), desc="img rank loop"):
        _i = _batch_img["clip_img"].cuda()
        _iid = _batch_img["img_index"]
        # _img_label = torch.tensor(_batch_img["img_label"])
        _iid = torch.tensor(_iid)

        out = pl_module.img_embeds(_i)
        feature = out["image_feats"]
        # img_cls = out["cls_img"]


        img = feature
        _l, _ = img.shape

        img_feats[i*64:i*64+_l] = img
        # img_feats_label[i*64:i*64+_l] = _img_label

        rank_iids[i*64:i*64+_l] = _iid

    torch.distributed.barrier()


    img_feats = img_feats / img_feats.norm(dim=1, keepdim=True)
    txt_feats = txt_feats/ txt_feats.norm(dim=1, keepdim=True)


    gather_img_features = img_feats
    gather_img_features_visual = torch.cat(all_gather(img_feats_visual))
    gather_txt_features = txt_feats
    gather_rank_iids = rank_iids



    iids = gather_rank_iids

    current_epoch = pl_module.hparams.config["current_epoch"]
    save_image_path = pl_module.hparams.config["save_image_path"]

    if os.path.exists(save_image_path):
        pass
    else:
        os.makedirs(save_image_path)


    logit_scale = pl_module.model.logit_scale.exp().to(gather_img_features.device)
    scores = logit_scale * gather_img_features @ gather_txt_features.t()

    # with open(json_path, 'r') as f:
    #     doc = json.load(f)
    #     information = doc["images"]
    #
    #
    # test_information = {}
    # for i in range(len(information)):
    #     if information[i]["split"] == "test":
    #         file_name = information[i]["filename"]
    #         text = information[i]["sentences"]
    #         test_information[file_name] = []
    #         for j in range(len(text)):
    #             raws = text[j]["raw"]
    #             test_information[file_name].append(raws)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    # all_information_text_retrieval_10 = path_text(topk10.indices, img_path_id, need_text_all, test_information, img_class_id)
    # all_information_text_retrieval_5 = path_text(topk5.indices, img_path_id, need_text_all, test_information, img_class_id)
    # all_information_text_retrieval_1 = path_text(topk1.indices, img_path_id, need_text_all, test_information, img_class_id)


    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()


    rep_texi_id = torch.tensor(rep_texi_id)
    rep_topk10_iids = rep_texi_id[topk10.indices]
    rep_topk5_iids = rep_texi_id[topk5.indices]
    rep_topk1_iids = rep_texi_id[topk1.indices]


    ttr_top10 = tr_top_k(iids.unsqueeze(0), text_correspondence, rep_topk10_iids, 10)
    ttr_top5 = tr_top_k(iids.unsqueeze(0), text_correspondence, rep_topk5_iids, 5)
    ttr_top1 = tr_top_k(iids.unsqueeze(0), text_correspondence, rep_topk1_iids, 1)

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]


    # #
    # # # print("this is the shape of topk:{}/{}".format(len(topk10.indices), len(topk10.indices[0])))
    # all_information_img_retrieval_10 = path_img(topk10.indices, img_path_id, need_text_all, test_information, text_class_id)
    # all_information_img_retrieval_5 = path_img(topk5.indices, img_path_id, need_text_all, test_information, text_class_id)
    # all_information_img_retrieval_1 = path_img(topk1.indices, img_path_id, need_text_all, test_information, text_class_id)
    # # # # print(all_information_img_retrieval_10)

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    iir_top10 = ir_top_k(rep_texi_id.unsqueeze(1), text_correspondence, topk10_iids, 10)
    iir_top5 = ir_top_k(rep_texi_id.unsqueeze(1), text_correspondence, topk5_iids, 5)
    iir_top1 = ir_top_k(rep_texi_id.unsqueeze(1), text_correspondence, topk1_iids, 1)


    a = sum([ttr_top1, ttr_top5, ttr_top10, iir_top1, iir_top5, iir_top10])/6

    result = {"iir_top1":round(iir_top1*100, 2),
              "iir_top5":round(iir_top5*100, 2),
              "iir_top10":round(iir_top10*100, 2),
              "ttr_top1":round(ttr_top1*100, 2),
              "ttr_top5":round(ttr_top5*100, 2),
              "ttr_top10":round(ttr_top10*100, 2),
              "mean":round(a*100, 2)
    }


    return (ir_r1*100, ir_r5*100, ir_r10*100, tr_r1*100, tr_r5*100, tr_r10*100), result


#
# def path_text(top_index, img, text, test_information, img_class_id):
#
#     number_to_class = {v: 0 for k, v in enumerate(set(img_class_id))}
#
#     # print(number_to_class)
#     batch, k = len(top_index), len(top_index[0])
#     categories_information = {
#         "text_retrieval_top":len(top_index[0]),
#         "all_items":batch,
#         "top_information":number_to_class,
#     }
#     all_information = list()
#
#     for i in range(batch):
#         img_path = img[i]
#         top_list = list()
#         position = list()
#         top_text = list()
#         for j in range(k):
#             index = top_index[i][j]
#             top_list.append(text[index])
#         img_path_text = (img_path, top_list)
#         img_ids = str(img_path[0][0]).split('/')[-1]
#         for key, val in test_information.items():
#             if img_ids == key:
#                 items = 0
#                 list_text = []
#                 flag = 0
#                 for j in range(len(top_list)):
#                     if top_list[j][0][0] in val:
#
#                         list_text.append(top_list[j][0][0])
#                         items +=1
#                         position.append(j)
#                         image_label = img_class_id[i],
#                         if flag !=1:
#                             categories_information["top_information"][image_label[0]] +=1
#                         flag = 1
#
#                 for h in range(len(position)):
#                     top_text.append([position[h], list_text[h]])
#                 image_label = img_class_id[i],
#                 top_position_information = {
#                     "image_ids":img_ids,
#                     "image_label":image_label,
#                     "top_position":position,
#                     "sum_number":items,
#                     "texts":top_text,
#                 }
#
#                 all_information.append(top_position_information)
#                 # print("###############################################")
#     print("this is text retrieval top {} categories_information:{}\n".format(k, categories_information))
#     all_information = pd.DataFrame(all_information)
#     all_information.to_excel("i2t_retrieval.xlsx", index=False)
#
#     return all_information
#
# def path_img(top_index, img, text, test_information, text_class_id):
#
#     number_to_class = {v: 0 for k, v in enumerate(set(text_class_id))}
#
#     top_index = top_index.permute(1, 0)
#     batch, k, = len(top_index), len(top_index[0])
#     categories_information = {
#         "top":len(top_index[0]),
#         "all_items":batch,
#         "top_information":number_to_class,
#     }
#     all_information = list()
#     for i in range(batch):
#         img_collection = []
#         text_item = text[i]
#         # print(text_item)
#         # top_list = list()
#         for j in range(k):
#             index = top_index[i][j]
#             img_path = str(img[index][0][0]).split('/')[-1]
#             img_collection.append(img_path)
#
#         img_id_ = []
#         flag=0
#         text_label = text_class_id[i]
#         for key, val in test_information.items():
#             if text_item[0][0] in val:
#                 img_id_.append(key)
#
#         item = 0
#         list_img = []
#         img_top_ids = []
#         for h in range(k):
#             if img_collection[h] in img_id_:
#                 list_img.append(h)
#                 item+=1
#                 img_top_ids.append(img_collection[h])
#                 if flag!=1:
#                     categories_information["top_information"][text_label] += 1
#                 flag=1
#             text_label = text_class_id[i]
#             top_position_information = {
#                 "text": text_item[0][0],
#                 "text_label": text_label,
#                 "top_position": list_img,
#                 "sum_number": item,
#                 "img_ids": img_top_ids,
#             }
#             all_information.append(top_position_information)
#
#     print("this is image retrieval top {} categories_information:{}\n".format(k, categories_information))
#
#     return all_information


def ir_top_k(text_idx, relate, img_idx, k):

    img_idx = img_idx.permute(1, 0)
    batch , _= img_idx.shape
    img_idx = img_idx[:, :k]
    top_sum = 0
    for i in range(batch):
        for j in range(k):
            img_index = img_idx[i][j]
            # print(img_index , "1")
            all_relation_index = text_idx[i]
            all_relation_index = "{}".format(all_relation_index.item())
            # print(all_relation_index, "@")
            if img_index in relate[all_relation_index]:
                top_sum += 1
                break
    return top_sum/batch

def tr_top_k(img_idx, relate, idx, k):
    text_idx = idx[:, :k]
    batch, _ = text_idx.shape
    top_sum = 0
    for i in range(batch):
        flag = 0
        for j in range(k):
            if flag == 0:
                rep_text = text_idx[i][j]
                all_relation_index = "{}".format(rep_text.item())
                all_relation = relate[all_relation_index]
                # print(i, rep_text, all_relation, all_relation_index)
                if img_idx[0, i] in all_relation:
                    top_sum += 1
                    flag=1

    return top_sum/batch


def compute_irtr(pl_module, batch):
    is_training_phase = pl_module.training

    _bs, _c, _h, _w = batch["image"][0].shape
    false_len = pl_module.hparams.config["draw_false_text"]
    text_ids = torch.stack(
        [batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1
    )
    text_masks = torch.stack(
        [batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1
    )
    text_labels = torch.stack(
        [batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1
    )

    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
    images = batch["image"][0].unsqueeze(1).expand(_bs, false_len + 1, _c, _h, _w)

    infer = pl_module.infer(
        {
            "image": [rearrange(images, "bs fs c h w -> (bs fs) c h w")],
            "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
            "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
            "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
        }
    )
    score = pl_module.rank_output(infer["cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    answer = torch.zeros(_bs).to(score).long()
    irtr_loss = F.cross_entropy(score, answer)

    ret = {
        "irtr_loss": irtr_loss,
    }

    phase = "train" if pl_module.training else "val"
    irtr_loss = getattr(pl_module, f"{phase}_irtr_loss")(ret["irtr_loss"])

    pl_module.log(f"irtr/{phase}/irtr_loss", irtr_loss)

    return ret


@torch.no_grad()
def compute_irtr_recall(pl_module):
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=64,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
        image_only=True
    )
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        text_preload.append(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
                "img_index": _b["img_index"],
            }
        )

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        (ie, im, _, _) = pl_module.transformer.visual_embed(
            _b["image"][0].to(pl_module.device),
            max_image_len=pl_module.hparams.config["max_image_len"],
            mask_it=False,
        )
        image_preload.append((ie, im, _b["img_index"][0]))

    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm.tqdm(image_preload, desc="rank loop"):
        _ie, _im, _iid = img_batch
        _, l, c = _ie.shape

        img_batch_score = list()
        for txt_batch in text_preload:
            fblen = len(txt_batch["text_ids"])
            ie = _ie.expand(fblen, l, c)
            im = _im.expand(fblen, l)

            with torch.cuda.amp.autocast():
                score = pl_module.rank_output(
                    pl_module.infer(
                        {
                            "text_ids": txt_batch["text_ids"],
                            "text_masks": txt_batch["text_masks"],
                            "text_labels": txt_batch["text_labels"],
                        },
                        image_embeds=ie,
                        image_masks=im,
                    )["cls_feats"]
                )[:, 0]

            img_batch_score.append(score)

        img_batch_score = torch.cat(img_batch_score)
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids.append(_iid)

    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()


    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)







def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def vqa_test_step(pl_module, batch, output):
    id2answer = (
        pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
        if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
        else pl_module.trainer.datamodule.dm_dicts["vqa"].id2answer
    )
    vqa_logits = output["vqa_logits"]
    vqa_preds = vqa_logits.argmax(dim=-1)
    vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
    questions = batch["text"]
    qids = batch["qid"]
    return {"qids": qids, "preds": vqa_preds}


def arc_test_step(pl_module, batch, output):
    return output


def vqa_test_wrapup(outs, model_name):
    rank = torch.distributed.get_rank()
    qids, preds = list(), list()
    for out in outs:
        qids += out["qids"]
        preds += out["preds"]

    rets = list()
    for qid, pred in zip(qids, preds):
        rets.append({"question_id": qid, "answer": pred})
    with open(f"vqa_submit_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob("vqa_submit_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result", exist_ok=True)
        with open(f"result/vqa_submit_{model_name}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"vqa_submit_{rank}.json")


def arc_test_wrapup(outs, caplen, model_name):
    rank = torch.distributed.get_rank()
    iids, captions = list(), list()
    for out in outs:
        iids += out["iid"]
        captions += out["captions"]

    rets = list()
    for iid, caption in zip(iids, captions):
        rets.append({"image_id": iid, "caption": caption})
    with open(f"coco_cap_len{caplen}_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob(f"coco_cap_len{caplen}_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result/arc", exist_ok=True)
        jsons = sorted(jsons, key=lambda x: x["image_id"])
        with open(f"result/arc/coco_cap_{model_name}_len{caplen}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"coco_cap_len{caplen}_{rank}.json")
