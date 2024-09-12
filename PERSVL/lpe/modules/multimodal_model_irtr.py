import os
import math
import torch
import numpy as np

from torch import nn
import torch.nn.functional as F

from lpe.modules.beit_module import BeitModel
from lpe.modules.lpe_module import LPEBuffer
from lpe.modules.lpe_utils import mlp
from lpe.modules.bert_module import BertForMaskedLM, BertConfig, BertModel

from transformers import BeitConfig

from lpe.modules.swin_transformer_module import SwinTransformerForSimMIM, SimMIM
from lpe.utils.utils import load_model


class RelationNet(nn.Module):
    def __init__(self, embed_dim, q_dim, kv_dim):
        super(RelationNet, self).__init__()
        self.embed_dim = embed_dim
        self.q_dim = q_dim
        self.kv_dim = kv_dim

        self.q_proj = nn.Linear(q_dim, self.embed_dim)
        self.k_proj = nn.Linear(kv_dim, self.embed_dim)
        self.v_proj = nn.Linear(kv_dim, self.embed_dim)

        self.dense = nn.Linear(self.embed_dim, self.embed_dim)
        # self.mlp = mlp(self.embed_dim, self.embed_dim, self.embed_dim, 2)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)

        self.layerNorm_q = nn.LayerNorm(q_dim)
        self.layerNorm_kv = nn.LayerNorm(kv_dim)

    def forward(self, q_feat, kv_feat, residual=True):
        # if pass_grad:
        #     q_value = self.q_proj(q_feat)
        #     k_value = self.k_proj(kv_feat)
        # else:
        #     q_value = self.q_proj(q_feat.detach())
        #     k_value = self.k_proj(kv_feat.detach())
        # q_value = self.q_proj(q_feat.detach())
        # k_value = self.k_proj(kv_feat.detach())
        original_q = q_feat

        if self.q_dim == self.kv_dim:
            q_feat = self.layerNorm_q(q_feat)
            kv_feat = self.layerNorm_q(kv_feat)
        else:
            q_feat = self.layerNorm_q(q_feat)
            kv_feat = self.layerNorm_kv(kv_feat)

        q_value = self.q_proj(q_feat)
        k_value = self.k_proj(kv_feat)
        v_value = self.v_proj(kv_feat)

        attention_sim = self.softmax(torch.matmul(q_value, k_value.transpose(-1, -2).contiguous()))
        attention_sim = self.dropout(attention_sim)
        attention_context = torch.matmul(attention_sim, v_value)

        hidden_states = self.dense(attention_context)
        hidden_states = self.dropout(hidden_states)

        if residual:
            output = original_q + hidden_states
        else:
            output = hidden_states
        # multiscale_feat = self.mlp(fusion_hl_feat)
        # multiscale_feat = self.dropout(multiscale_feat)
        # multiscale_feat = self.layerNorm(multiscale_feat + fusion_hl_feat)

        return output


class HCN(nn.Module):
    def __init__(self, high_level_dim: int, low_level_dim_list: list, num_low_level_use: int):
        super(HCN, self).__init__()
        self.layers = nn.ModuleList([
            RelationNet(
                embed_dim=high_level_dim,
                q_dim=high_level_dim,
                kv_dim=low_level_dim_list[i]
            )
            for i in range(len(low_level_dim_list))
        ])

        self.num_low_level_use = num_low_level_use

    def forward(self, img_feats):
        q_feat = img_feats[-1]

        for i in range(len(self.layers) - 1, -1 + 4 - self.num_low_level_use, -1):
            layer_module = self.layers[i]
            q_feat = layer_module(q_feat=q_feat, kv_feat=img_feats[i])

        return q_feat


class VisualFusionMIM(nn.Module):
    def __init__(self, args, encoder_stride, in_chans, patch_size):
        super(VisualFusionMIM, self).__init__()
        self.encoder_stride = encoder_stride

        beit_config = BeitConfig.from_json_file(os.path.join(args.modify_beit_config, 'config_beit.json'))
        self.visual_fusion_encoder = BeitModel.from_pretrained(args.beit_encoder, config=beit_config)
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.visual_fusion_encoder.config.hidden_size,
                out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )

        self.in_chans = in_chans
        self.patch_size = patch_size

    def forward(self, image_embeds, text_embeds, image, mask):
        output = self.visual_fusion_encoder(
            encoder_embeds=image_embeds,
            encoder_hidden_states=text_embeds,
            return_dict=True,
            mode='fusion',
        )
        img_output = output.last_hidden_state[:, 1:, :]
        B, L, C = img_output.shape
        H = W = int(L ** 0.5)

        x = img_output.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        img_rec = self.decoder(x)

        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(
            1).contiguous().to(x.device)
        loss_recon = F.l1_loss(image, img_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        return loss


class CRSRT(nn.Module):
    """Cross-modal RS Image-Text Retrieval model based on Transformer"""
    def __init__(self, args, swin_cfg, tokenizer):
        super(CRSRT, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.mlm_probability = args.mlm_probability

        bert_config = BertConfig.from_json_file(os.path.join(args.modify_bert_config, 'config_bert.json'))
        self.text_encoder = BertForMaskedLM.from_pretrained(args.text_encoder, config=bert_config)
        self.visual_encoder = SwinTransformerForSimMIM(swin_cfg)

        self.hcn_net = HCN(
            high_level_dim=args.high_level_dim,
            low_level_dim_list=args.low_level_dim_list,
            num_low_level_use=args.num_low_level_use,
        )

        self.visual_fusion_net = VisualFusionMIM(
            args, args.encoder_stride, self.visual_encoder.in_chans, self.visual_encoder.patch_size)

        visual_checkpoint = torch.load(args.visual_encoder, map_location='cpu')
        visual_state_dict = load_model(visual_checkpoint, self.visual_encoder, 'model')
        self.visual_encoder.load_state_dict(visual_state_dict)

        visual_embed_dim = args.high_level_dim
        text_embed_dim = self.text_encoder.config.hidden_size
        proj_dim = args.embed_dim
        self.temp = nn.Parameter(torch.ones([]) * args.temp)

        self.vision_proj = nn.Linear(visual_embed_dim, proj_dim)
        self.text_proj = nn.Linear(text_embed_dim, proj_dim)

        self.fusion_net = RelationNet(
            embed_dim=text_embed_dim,
            q_dim=text_embed_dim,
            kv_dim=text_embed_dim
        )

        self.fusion_query = nn.Parameter(torch.rand((1, text_embed_dim)), requires_grad=True)
        self.beta_rate = nn.Parameter(torch.tensor(args.beta_rate), requires_grad=True)   # 0.05
        self.itm_head = nn.Linear(text_embed_dim, 2)

        self.lpe_buffer = LPEBuffer(args, text_embed_dim)

    def forward(self, image, text, idx_mask, noun_idxs=None, img_mask=None, orgin_text=None,
                epoch=None, model_m=None, use_experience=False):
        image_embeds, image_atts = self.get_img_embeds(image)
        text_embeds = self.get_txt_embeds(text)

        # =============================================================
        # task 1: image-text contrastive learning task
        # idx = idx.view(-1, 1)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        # pos_idx = torch.eq(idx, idx.t()).float()
        pos_idx = idx_mask.float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        sim_i2t = image_feat @ text_feat.t() / self.temp
        sim_t2i = text_feat @ image_feat.t() / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()

        loss_itc = (loss_i2t + loss_t2i) / 2

        # =============================================================
        # task 2: image-text matching task

        # # forward the positive image-text pair
        # output_pos = self.text_encoder.bert(
        #     encoder_embeds=text_embeds,
        #     attention_mask=text.attention_mask,
        #     encoder_hidden_states=image_embeds,
        #     encoder_attention_mask=image_atts,
        #     return_dict=True,
        #     mode='fusion',
        # )

        # if epoch + 1 >= self.args.start_exp_epoch:
        #     self.args.use_alter_mlm = False
        #     self.args.use_mlm = False
        #     self.args.use_mim = False

        # step 1: hard negative sampling (hns) and typical negative sampling (tns)
        with torch.no_grad():
            bs = image.size(0)
            hns_weights_i2t = F.softmax(sim_i2t[:, :bs] + 1e-4, dim=1)
            hns_weights_t2i = F.softmax(sim_t2i[:, :bs] + 1e-4, dim=1)
            tns_weights_i2t = F.softmax(-sim_i2t[:, :bs] + 1e-4, dim=1)
            tns_weights_t2i = F.softmax(-sim_t2i[:, :bs] + 1e-4, dim=1)

            mask = idx_mask
            hns_weights_i2t.masked_fill_(mask, 0)
            hns_weights_t2i.masked_fill_(mask, 0)
            tns_weights_i2t.masked_fill_(mask, 0)
            tns_weights_t2i.masked_fill_(mask, 0)

            # select a negative image for each text
            image_embeds_neg = []
            image_embeds_tns = []
            image_idxs_neg = []
            image_idxs_tns = []
            for b in range(bs):
                neg_idx = torch.multinomial(hns_weights_t2i[b], 1).item()
                tns_idx = torch.multinomial(tns_weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds[neg_idx])
                image_embeds_tns.append(image_embeds[tns_idx])
                image_idxs_neg.append(neg_idx)
                image_idxs_tns.append(tns_idx)
            image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
            image_embeds_tns = torch.stack(image_embeds_tns, dim=0)

            # select a negative text for each image
            text_embeds_neg = []
            text_atts_neg = []
            text_embeds_tns = []
            text_atts_tns = []
            text_idxs_neg = []
            text_idxs_tns = []
            for b in range(bs):
                neg_idx = torch.multinomial(hns_weights_i2t[b], 1).item()
                tns_idx = torch.multinomial(tns_weights_i2t[b], 1).item()

                text_embeds_neg.append(text_embeds[neg_idx])
                text_atts_neg.append(text.attention_mask[neg_idx])

                text_embeds_tns.append(text_embeds[tns_idx])
                text_atts_tns.append(text.attention_mask[tns_idx])

                text_idxs_neg.append(neg_idx)
                text_idxs_tns.append(tns_idx)
            text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)
            text_embeds_tns = torch.stack(text_embeds_tns, dim=0)
            text_atts_tns = torch.stack(text_atts_tns, dim=0)

            text_embeds_all = torch.cat([text_embeds, text_embeds, text_embeds_neg], dim=0)
            text_atts_all = torch.cat([text.attention_mask, text.attention_mask, text_atts_neg], dim=0)

            image_embeds_all = torch.cat([image_embeds, image_embeds_neg, image_embeds], dim=0)
            image_atts_all = torch.cat([image_atts, image_atts, image_atts], dim=0)
            # text_embeds_all = torch.cat([text_embeds, text_embeds, text_embeds_neg, text_embeds, text_embeds_tns], dim=0)
            # text_atts_all = torch.cat(
            #     [text.attention_mask, text.attention_mask, text_atts_neg, text.attention_mask, text_atts_tns], dim=0)
            #
            # image_embeds_all = torch.cat([image_embeds, image_embeds_neg, image_embeds, image_embeds_tns, image_embeds], dim=0)
            # image_atts_all = torch.cat([image_atts, image_atts, image_atts, image_atts, image_atts], dim=0)

        # step 2: forward all pair
        vl_embeddings = self.fusion_forward(
            text_embeds=text_embeds_all,
            text_attn=text_atts_all,
            image_embeds=image_embeds_all,
            image_attn=image_atts_all
        )

        vl_output = self.itm_head(vl_embeddings)
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).unsqueeze(1).to(image.device)
        itm_targets = torch.cat([1 - itm_labels, itm_labels], dim=1)
        itm_score = F.log_softmax(vl_output, dim=1) * itm_targets
        loss_itm = -torch.sum(itm_score, dim=1).mean()

        if use_experience:
            # step3: momentum forward
            vl_embeddings_m, score = model_m.m_forward(
                image, text, image_idxs_neg, text_idxs_neg, image_idxs_tns, text_idxs_tns)
            # vl_output_all = self.itm_head(vl_embeddings_all)
            alter_loss_itm, beta_loss_score, beta, weight, no_alter_num = self.calculate_alter_itm_loss(
                vl_output=vl_output,
                itm_labels=itm_labels,
                vl_embeddings=vl_embeddings_m,
                bs=image.shape[0], epoch=epoch,
                itm_head=model_m.itm_head, score=score
            )
            beta_loss = F.l1_loss(itm_score, beta_loss_score)
        else:
            alter_loss_itm = loss_itm
            beta = torch.tensor(0.)
            weight = torch.tensor(0.)
            no_alter_num = torch.tensor(0.)
            beta_loss = torch.tensor(0.)

        # =============================================================
        # task 3: Mask Language Modeling task
        if self.args.use_mlm:
            input_ids = text.input_ids.clone()
            labels = input_ids.clone()

            if noun_idxs is not None:
                probability_matrix = torch.full(labels.shape, 0.0)

                for each, noun_idx in enumerate(noun_idxs):
                    current_mlm_prob = min(self.mlm_probability * input_ids.shape[1] / len(noun_idx), 0.9)
                    probability_matrix[each, noun_idx] = current_mlm_prob
            else:
                probability_matrix = torch.full(labels.shape, self.mlm_probability)

            input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, image.device, targets=labels,
                                          probability_matrix=probability_matrix)

            mlm_output = self.text_encoder(input_ids,
                                           attention_mask=text.attention_mask,
                                           encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_atts,
                                           return_dict=True,
                                           labels=labels,
                                           )
            loss_mlm = mlm_output.loss
        else:
            loss_mlm = torch.tensor(0.)

        # =============================================================
        # task 4: Mask Image Modeling task
        if self.args.use_mim:
            assert img_mask is not None
            loss_mim = self.visual_fusion_net(image_embeds, text_embeds, image, img_mask)
        else:
            loss_mim = torch.tensor(0.)

        return loss_itm, alter_loss_itm, loss_itc, loss_mlm, loss_mim, beta_loss, beta, weight, no_alter_num

    def get_txt_embeds(self, text):
        text_output = self.text_encoder.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True, mode='text')
        return text_output.last_hidden_state

    def get_img_embeds(self, image, mask=None, detail=False):
        visual_output = self.visual_encoder(image, mask)

        if self.args.use_multiscale_visual_fusion:
            image_embeds = self.hcn_net(visual_output['stage_last_feat'])
        else:
            image_embeds = visual_output['stage_last_feat'][-1]

        cls_embedding = image_embeds.mean(1).unsqueeze(1)
        image_embeds = torch.cat([cls_embedding, image_embeds], dim=1)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        if detail:
            return image_embeds, image_atts, None, None
        else:
            return image_embeds, image_atts

    def fusion_forward(self, text_embeds, text_attn, image_embeds, image_attn):
        vl_embeddings_i2t, vl_embeddings_t2i = None, None

        if self.args.use_mlm or (self.args.use_mlm is False and self.args.use_mim is False):
            output_i2t = self.text_encoder.bert(
                encoder_embeds=text_embeds,
                attention_mask=text_attn,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attn,
                return_dict=True,
                mode='fusion',
            )
            vl_embeddings_i2t = output_i2t.last_hidden_state[:, 0, :]

        if self.args.use_mim or (self.args.use_mlm is False and self.args.use_mim is False):
            output_t2i = self.visual_fusion_net.visual_fusion_encoder(
                encoder_embeds=image_embeds,
                encoder_hidden_states=text_embeds,
                return_dict=True,
                mode='fusion',
            )
            vl_embeddings_t2i = output_t2i.last_hidden_state[:, 0, :]

        assert vl_embeddings_t2i is not None or vl_embeddings_i2t is not None

        if vl_embeddings_i2t is not None and vl_embeddings_t2i is not None:
            intermediate_embeddings = torch.stack([vl_embeddings_i2t, vl_embeddings_t2i], dim=1)
            if self.args.use_mm_fusion_query:
                batch_query = self.fusion_query.repeat(intermediate_embeddings.shape[0], 1, 1)
                vl_embeddings = self.fusion_net(q_feat=batch_query, kv_feat=intermediate_embeddings).squeeze(1)
            else:
                vl_embeddings = self.fusion_net(q_feat=intermediate_embeddings, kv_feat=intermediate_embeddings)
                vl_embeddings = vl_embeddings.mean(1)
                # vl_embeddings = intermediate_embeddings.mean(1)
        elif vl_embeddings_i2t is None:
            vl_embeddings = vl_embeddings_t2i
        else:
            vl_embeddings = vl_embeddings_i2t

        return vl_embeddings

    @torch.no_grad()
    def m_forward(self, image, text, img_neg_idxs, txt_neg_idxs, img_idxs_tns, txt_idxs_tns):
        """used for momentum forward"""
        image_embeds, image_atts = self.get_img_embeds(image)
        text_embeds = self.get_txt_embeds(text)

        image_embeds_neg = torch.stack([image_embeds[idx] for idx in img_neg_idxs], dim=0)
        image_embeds_tns = torch.stack([image_embeds[idx] for idx in img_idxs_tns], dim=0)

        text_embeds_neg = torch.stack([text_embeds[idx] for idx in txt_neg_idxs], dim=0)
        text_attns_neg = torch.stack([text.attention_mask[idx] for idx in txt_neg_idxs], dim=0)
        text_embeds_tns = torch.stack([text_embeds[idx] for idx in txt_idxs_tns], dim=0)
        text_attns_tns = torch.stack([text.attention_mask[idx] for idx in txt_idxs_tns], dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds, text_embeds_neg, text_embeds, text_embeds_tns], dim=0)
        text_atts_all = torch.cat(
            [text.attention_mask, text.attention_mask, text_attns_neg, text.attention_mask, text_attns_tns], dim=0)
        # text_embeds_all = torch.cat([text_embeds, text_embeds, text_embeds_neg], dim=0)
        # text_atts_all = torch.cat([text.attention_mask, text.attention_mask, text_attns_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds, image_embeds_neg, image_embeds, image_embeds_tns, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts, image_atts, image_atts, image_atts], dim=0)
        # image_embeds_all = torch.cat([image_embeds, image_embeds_neg, image_embeds], dim=0)
        # image_atts_all = torch.cat([image_atts, image_atts, image_atts], dim=0)

        vl_embeddings = self.fusion_forward(
            text_embeds=text_embeds_all,
            text_attn=text_atts_all,
            image_embeds=image_embeds_all,
            image_attn=image_atts_all
        )

        vl_output = F.softmax(self.itm_head(vl_embeddings), dim=1)

        return vl_embeddings, vl_output[:, 1]

    def calculate_alter_itm_loss(self, vl_output, itm_labels, vl_embeddings, bs, epoch, itm_head, score):
        itm_labels_all = torch.cat(
            [itm_labels.squeeze(1), torch.zeros(2 * bs, dtype=torch.float).to(itm_labels.device)], dim=0).unsqueeze(1)
        itm_targets = torch.cat([1 - itm_labels, itm_labels], dim=1)

        neg_alter_label = torch.clip(itm_labels_all - 1, max=0)
        itm_alter_label = itm_labels_all + neg_alter_label
        itm_alter_label[bs: 3*bs] = torch.zeros(2 * bs, dtype=torch.float).to(itm_labels.device).unsqueeze(1)

        exp_itm_label, exp_wo_alter_label, gamma, weight = self.lpe_buffer.get_exp_targets(
            bs=bs, epoch=epoch,
            itm_alter_label=itm_alter_label,
            itm_head=itm_head, score=score, vl_embeddings=vl_embeddings
        )
        beta = gamma * self.args.beta * (weight ** (torch.exp(self.beta_rate * 10) + 1))
        target_beta = torch.cat([beta.unsqueeze(1), beta.unsqueeze(1)], dim=1)

        exp_itm_target = torch.cat([1 - exp_itm_label, exp_itm_label], dim=1)
        alter_itm_target = target_beta * exp_itm_target + (1 - target_beta) * itm_targets
        norm_factor = torch.sum(alter_itm_target, dim=1).unsqueeze(1)
        alter_itm_target /= torch.cat([norm_factor, norm_factor], dim=1)

        # exp_wo_alter_label = exp_wo_alter_label.squeeze(1)
        # exp_wo_alter_label[(exp_wo_alter_label > -0.5) & (exp_wo_alter_label < 0.0)] = -0.5
        # exp_wo_alter_label = exp_wo_alter_label.unsqueeze(1)
        # itm_alter_label[bs: 3*bs] = exp_wo_alter_label[bs:] * beta

        push_embedding = vl_embeddings
        push_label = itm_alter_label

        self.lpe_buffer.exp_dequeue_and_enqueue(vl_feat=push_embedding, label=push_label)

        alter_loss_itm = -torch.sum(F.log_softmax(vl_output, dim=1) * alter_itm_target, dim=1).mean()
        beta_loss_score = F.log_softmax(vl_output.detach(), dim=1) * alter_itm_target

        return alter_loss_itm, beta_loss_score, beta[bs:].mean(), weight[bs:].mean(), \
               exp_itm_label.squeeze(1).cpu().numpy().tolist().count(0.0)

    @torch.no_grad()
    def direct_forward(self, image, text):
        image_embeds, image_atts = self.get_img_embeds(image)
        text_embeds = self.get_txt_embeds(text)

        output = self.fusion_forward(
            text_embeds=text_embeds,
            text_attn=text.attention_mask,
            image_embeds=image_embeds,
            image_attn=image_atts
        )

        return output

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids


@torch.no_grad()
def momentum_update(model, model_m, momentum_coeff):
    for param, param_m in zip(model.parameters(), model_m.parameters()):
        param_m.data = param_m.data * momentum_coeff + param.data * (1. - momentum_coeff)

