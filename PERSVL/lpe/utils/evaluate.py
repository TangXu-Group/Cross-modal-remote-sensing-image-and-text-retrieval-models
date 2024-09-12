import time
import datetime
import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .utils import MetricLogger, get_world_size, get_rank


@torch.no_grad()
def evaluation(args, model, data_loader, tokenizer, device):
    # test
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    print('Computing features for evaluation...')
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_feats = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(
            device)
        text_feat = model.get_txt_embeds(text_input)
        text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]))
        text_embeds.append(text_embed)
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)
    text_embeds = torch.cat(text_embeds, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    image_feats = []
    image_embeds = []
    for image, img_id in data_loader:
        image = image.to(device)
        image_feat, _ = model.get_img_embeds(image)
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)

        image_feats.append(image_feat)
        image_embeds.append(image_embed)

    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full((len(data_loader.dataset.image), len(texts)), -100.0).to(device)

    num_tasks = get_world_size()
    rank = get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        k_test = min(sims.shape[0], args.k_test)
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)

        encoder_output = image_feats[start + i].repeat(k_test, 1, 1)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)

        vl_embeddings = model.fusion_forward(
            text_embeds=text_feats[topk_idx],
            text_attn=text_atts[topk_idx],
            image_embeds=encoder_output,
            image_attn=encoder_att
        )

        score = model.itm_head(vl_embeddings)[:, 1]
        score_matrix_i2t[start + i, topk_idx] = score

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        k_test = min(sims.shape[0], args.k_test)
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        encoder_output = image_feats[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)

        vl_embeddings = model.fusion_forward(
            text_embeds=text_feats[start + i].repeat(k_test, 1, 1),
            text_attn=text_atts[start + i].repeat(k_test, 1),
            image_embeds=encoder_output,
            image_attn=encoder_att
        )

        score = model.itm_head(vl_embeddings)[:, 1]
        score_matrix_t2i[start + i, topk_idx] = score

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def evaluation_w_exp(args, model, data_loader, tokenizer, device):
    # test
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    print('Computing features for evaluation...')
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_feats = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(
            device)
        text_feat = model.get_txt_embeds(text_input)
        text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]))
        text_embeds.append(text_embed)
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)
    text_embeds = torch.cat(text_embeds, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    image_feats = []
    image_embeds = []
    for image, img_id in data_loader:
        image = image.to(device)
        image_feat, _ = model.get_img_embeds(image)
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)

        image_feats.append(image_feat)
        image_embeds.append(image_embed)

    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full((len(data_loader.dataset.image), len(texts)), -100.0).to(device)

    num_tasks = get_world_size()
    rank = get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        k_test = min(sims.shape[0], args.k_test)
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)

        encoder_output = image_feats[start + i].repeat(k_test, 1, 1)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)

        vl_embeddings = model.fusion_forward(
            text_embeds=text_feats[topk_idx],
            text_attn=text_atts[topk_idx],
            image_embeds=encoder_output,
            image_attn=encoder_att
        )

        output = model.itm_head(vl_embeddings)
        score = output[:, 1]

        exp_score, weight = model.lpe_buffer.get_eval_exp_scores(vl_embeddings, F.softmax(output, dim=1), model.itm_head)
        beta = args.beta * (weight**(torch.exp(model.beta_rate) + 1))
        final_score = beta * exp_score + (1 - beta) * score

        score_matrix_i2t[start + i, topk_idx] = final_score

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        k_test = min(sims.shape[0], args.k_test)
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        encoder_output = image_feats[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)

        vl_embeddings = model.fusion_forward(
            text_embeds=text_feats[start + i].repeat(k_test, 1, 1),
            text_attn=text_atts[start + i].repeat(k_test, 1),
            image_embeds=encoder_output,
            image_attn=encoder_att
        )

        output = model.itm_head(vl_embeddings)
        score = output[:, 1]

        exp_score, weight = model.lpe_buffer.get_eval_exp_scores(vl_embeddings, F.softmax(output, dim=1), model.itm_head)
        beta = args.beta * (weight**(torch.exp(model.beta_rate) + 1))
        final_score = beta * exp_score + (1 - beta) * score

        score_matrix_t2i[start + i, topk_idx] = final_score

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def evaluation_contrastive_only(args, model, data_loader, tokenizer, device):
    # test
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    print('Computing features for evaluation...')
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_feats = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(
            device)
        text_feat = model.get_txt_embeds(text_input)
        text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]))
        text_embeds.append(text_embed)
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)
    text_embeds = torch.cat(text_embeds, dim=0)

    image_feats = []
    image_embeds = []
    for image, img_id in data_loader:
        image = image.to(device)
        image_feat, _ = model.get_img_embeds(image)
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)

        image_feats.append(image_feat)
        image_embeds.append(image_embed)

    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix_i2t = image_embeds @ text_embeds.t()
    sims_matrix_t2i = sims_matrix_i2t.t()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    return sims_matrix_i2t.cpu().numpy(), sims_matrix_t2i.cpu().numpy()


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt, txt2id_dict, id2txt_dict, text):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            for pos in id2txt_dict[i]:
                tmp = np.where(inds == pos)[0][0]
                if tmp < rank:
                    rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        rank = 1e20
        for i in txt2img[txt2id_dict[text[index]]]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
    # for index, score in enumerate(scores_t2i):
    #     inds = np.argsort(score)[::-1]
    #     ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean}
    return eval_result


