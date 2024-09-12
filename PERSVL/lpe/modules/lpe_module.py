import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from lpe.modules.lpe_utils import concat_all_gather


class LPEBuffer(nn.Module):
    def __init__(self, args, feature_dim):
        super(LPEBuffer, self).__init__()
        self.args = args

        self.exp_queue_capacity = self.args.exp_queue_capacity
        self.similarity = nn.CosineSimilarity()

        self.register_buffer("experience_queue", torch.randn(self.exp_queue_capacity, feature_dim))
        self.register_buffer("exp_label_queue", torch.full((self.exp_queue_capacity, 1), 0.0))
        self.register_buffer("exp_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("is_full", torch.zeros(1, dtype=torch.long))
        self.register_buffer("is_empty", torch.ones(1, dtype=torch.long))

    @torch.no_grad()
    def get_exp_targets(self, bs, epoch, itm_alter_label, itm_head, score, vl_embeddings):
        """

        Args:
            bs: batch size
            epoch: current epoch
            itm_alter_label: alter of itm label, the range of it is in {-1, 1}
            itm_head: itm classification head
            vl_embeddings: vl embeddings
            score: current score calculating by momentum model

        Returns: exp_itm_target, gamma

        """
        if int(self.is_empty) == 0:
            pre_exps_embeds, pre_labels = self._exp_pop_queue()
            # pre_exps_embeds = torch.cat([vl_embeddings_m[: bs], vl_embeddings_m[bs:], pre_exps_embeds], dim=0)
            pre_exps_scores = F.softmax(itm_head(pre_exps_embeds), dim=1)[:, 1]
            pre_exps_scores = torch.cat([score, pre_exps_scores], dim=0)
            pre_exps_embeds = torch.cat([vl_embeddings, pre_exps_embeds], dim=0)
            pre_labels = torch.cat([itm_alter_label, pre_labels], dim=0)
            # pre_exps_scores = torch.cat([score[: bs], score[3*bs:], pre_exps_scores], dim=0)
            # pre_exps_embeds = torch.cat([vl_embeddings_m[: bs], vl_embeddings_m[3*bs:], pre_exps_embeds], dim=0)
            # pre_labels = torch.cat([itm_alter_label[: bs], itm_alter_label[3*bs:], pre_labels], dim=0)
        else:
            # pre_exps_embeds = torch.cat([vl_embeddings_m[: bs], vl_embeddings_m[bs:]], dim=0)
            # pre_exps_scores = torch.cat([score[: bs], score[3*bs:]], dim=0)
            # pre_exps_embeds = torch.cat([vl_embeddings_m[: bs], vl_embeddings_m[3*bs:]], dim=0)
            # pre_labels = torch.cat([itm_alter_label[: bs], itm_alter_label[3*bs:]], dim=0)
            pre_exps_scores = score
            pre_exps_embeds = vl_embeddings
            pre_labels = itm_alter_label

        exp_itm_label = []
        exp_wo_alter_label = []
        weights = []
        for i in range(3 * bs):
            if i >= bs: # i >= bs
                batch_feats = vl_embeddings[i].repeat(pre_exps_embeds.shape[0], 1)
                cos_sim = self.similarity(batch_feats, pre_exps_embeds)
                dist = (pre_exps_scores - score[i]) ** 2

                # sim_filter = (dist > self.args.dist_threshold**2) | (cos_sim < self.args.sim_threshold)
                # sim_filter = dist > self.args.dist_threshold ** 2
                # dist[dist > self.args.dist_threshold**2] = 1.
                # sim = -dist

                sim_filter = cos_sim < self.args.sim_threshold
                sim = cos_sim * (1 - dist)
                sim[sim_filter] = -1.

                distribution = (sim - torch.min(sim)) / (torch.max(sim) - torch.min(sim) + 1e-8)
                distribution /= torch.sum(distribution) + 1e-8

                tmp_wo_alter_label = torch.sum(distribution.unsqueeze(1) * pre_labels, dim=0)
                tmp_alter_label = torch.clip(tmp_wo_alter_label, min=0)

                sim[sim_filter] = 0.
                cnt = sim_filter.cpu().detach().numpy().tolist().count(False)
                if cnt != 0:
                    weights.append(torch.sum(sim) / cnt)
                else:
                    weights.append(torch.tensor(0.).to(tmp_wo_alter_label.device))
            else:
                tmp_alter_label = torch.tensor([1.]).to(score.device)
                tmp_wo_alter_label = tmp_alter_label
                weights.append(torch.tensor(0.).to(tmp_wo_alter_label.device))

            exp_itm_label.append(tmp_alter_label)
            exp_wo_alter_label.append(tmp_wo_alter_label)

        exp_itm_label = torch.stack(exp_itm_label, dim=0)
        exp_wo_alter_label = torch.stack(exp_wo_alter_label, dim=0)
        gamma = pre_exps_scores.shape[0] / (self.exp_queue_capacity + score.shape[0])

        weights = torch.stack(weights, dim=0)
        # weight = weights.mean()
        weights = torch.clip(weights - self.args.sim_threshold, min=0)
        weights /= 1 - self.args.sim_threshold

        # beta = self.balance_func(epoch + gama - self.args.start_exp_epoch, -self.args.cfd_epoch)

        return exp_itm_label, exp_wo_alter_label, gamma, weights

    @torch.no_grad()
    def get_eval_exp_scores(self, vl_embeddings, score, itm_head):
        assert int(self.is_empty) == 0
        pre_exps_embeds, _ = self._exp_pop_queue()
        # pre_exps_embeds = torch.cat([vl_embeddings_m[: bs], vl_embeddings_m[bs:], pre_exps_embeds], dim=0)
        pre_exps_scores = itm_head(pre_exps_embeds)
        pre_exps_relation_score = F.softmax(pre_exps_scores, dim=1)[:, 1]
        pre_exps_scores = pre_exps_scores[:, 1]

        exp_itm_scores = []
        weights = []
        for i in range(vl_embeddings.shape[0]):
            batch_feats = vl_embeddings[i].repeat(pre_exps_embeds.shape[0], 1)
            cos_sim = self.similarity(batch_feats, pre_exps_embeds)
            # dist = (pre_exps_relation_score - score[i]) ** 2

            sim_filter = cos_sim < self.args.sim_threshold
            sim = cos_sim
            sim[sim_filter] = -1.

            distribution = (sim - torch.min(sim)) / (torch.max(sim) - torch.min(sim) + 1e-8)
            distribution /= torch.sum(distribution) + 1e-8

            tmp_wo_alter_label = torch.sum(distribution * pre_exps_scores, dim=0)
            tmp_alter_label = torch.clip(tmp_wo_alter_label, min=0)

            cos_sim[sim_filter] = 0.
            cnt = sim_filter.cpu().detach().numpy().tolist().count(False)

            if cnt != 0:
                weights.append(torch.sum(cos_sim) / cnt)

            exp_itm_scores.append(tmp_wo_alter_label)

        exp_itm_scores = torch.stack(exp_itm_scores, dim=0)

        if weights != []:
            weights = torch.stack(weights, dim=0)
            weight = weights.mean()
            weight -= self.args.sim_threshold
            weight /= 1 - self.args.sim_threshold
        else:
            weight = torch.tensor(0.).to(exp_itm_scores.device)

        return exp_itm_scores, weight

    @torch.no_grad()
    def exp_dequeue_and_enqueue(self, vl_feat, label):
        # gather keys before updating queue
        if self.args.distributed:
            vl_feats = concat_all_gather(vl_feat, self.args)
            labels = concat_all_gather(label, self.args)
        else:
            vl_feats = vl_feat
            labels = label

        batch_size = vl_feats.shape[0]
        if batch_size > 0: self.is_empty[0] = 0

        ptr = int(self.exp_queue_ptr)

        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size <= self.exp_queue_capacity:
            self.experience_queue[ptr:ptr + batch_size, :] = vl_feats
            self.exp_label_queue[ptr:ptr + batch_size, :] = labels
        else:
            overflow = (ptr + batch_size) % self.exp_queue_capacity
            cut_ptr = batch_size - overflow

            self.experience_queue[ptr:, :] = vl_feats[: cut_ptr]
            self.exp_label_queue[ptr:, :] = labels[: cut_ptr]

            self.experience_queue[:overflow, :] = vl_feats[cut_ptr:]
            self.exp_label_queue[:overflow, :] = labels[cut_ptr:]

        pre_ptr = ptr
        ptr = (ptr + batch_size) % self.exp_queue_capacity  # move pointer

        if ptr < pre_ptr: self.is_full[0] = 1

        self.exp_queue_ptr[0] = ptr

    @torch.no_grad()
    def _exp_pop_queue(self):
        if int(self.is_full) == 1:
            return self.experience_queue.clone(), self.exp_label_queue.clone()
        else:
            ptr = int(self.exp_queue_ptr)
            return self.experience_queue[: ptr, :].clone(), self.exp_label_queue[: ptr, :].clone()

    def get_exp_queue_size(self):
        if self.is_full[0] == 1:
            return self.exp_queue_capacity
        else:
            return int(self.exp_queue_ptr)

    def balance_func(self, x, b=-15):
        return (1 + np.tanh((15 / -b)*x - 15)) / 2

        # return (1 + np.tanh(x + b)) / 2


class Similarity(nn.Module):
    def __init__(self):
        super(Similarity, self).__init__()
        self.cal_similarity = nn.CosineSimilarity()

    def forward(self, x, y):
        return self.cal_similarity(x, y)






