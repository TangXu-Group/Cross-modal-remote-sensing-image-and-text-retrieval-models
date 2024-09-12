import torch
import copy as cp
import torch.nn as nn
import numpy as np
from lpe.modules.lpe_utils import to_np


class TypicalExperience(object):
    def __init__(self, image_id, text, vl_embedding, pred_class, label, q_value):
        self.image_id = image_id
        self.text = text
        self.vl_embedding = vl_embedding
        self.pred_class = pred_class
        self.label = label
        self.q_value = q_value

    def __eq__(self, other):
        return self.image_id == other.image_id and self.text == other.text

    def __hash__(self):
        return hash(str(self.image_id) + self.text)

    def __str__(self):
        return str(self.image_id) + self.text


class ExperienceQueue(object):
    def __init__(self, args):
        self.args = args
        self.capacity = args.queue_capacity
        self.queue = [TypicalExperience(None, None, None, None, None, -1)] * self.capacity
        self.queue_q_list = []
        self.index_dict = {}
        self.index_list = []
        self.idx = 0
        self._idxs = None
        self.full = False
        self.empty = True

    def __len__(self):
        if self.full:
            return self.capacity
        else:
            return self.idx

    def _push(self, experiences: list, q_cur, pair_idxs):
        batch_size = len(experiences)
        if batch_size > 0:
            self.empty = False
        if self.idx + batch_size <= self.capacity:
            self.queue[self.idx: self.idx + batch_size] = cp.copy(experiences)
            self.queue_q_list[self.idx: self.idx + batch_size] = q_cur
        else:
            overflow = (self.idx + batch_size) % self.capacity
            cut_idx = batch_size - overflow

            self.queue[self.idx:] = cp.copy(experiences[: cut_idx])
            self.queue_q_list[self.idx:] = q_cur[: cut_idx]

            self.queue[: overflow] = cp.copy(experiences[cut_idx:])
            self.queue_q_list[: overflow] = q_cur[cut_idx:]

        prev_idx = self.idx
        self.idx = (self.idx + batch_size) % self.capacity
        self.full = self.full or self.idx < prev_idx

    def _push_while_full(self, size, pair_idxs, experiences, q_cur, start_idx=0):
        choice_prob = torch.softmax(-torch.tensor(self.queue_q_list), dim=0)
        replace_idxs = np.random.choice(range(self.capacity), size=size, replace=False, p=choice_prob.numpy())
        for i in range(len(replace_idxs)):
            if pair_idxs[start_idx + i] in self.index_list:
                # avoid overlapping data
                tmp_idx = self.index_dict[pair_idxs[start_idx + i]]
                self.queue[tmp_idx] = cp.copy(experiences[start_idx + i])
                self.queue_q_list[tmp_idx] = q_cur[start_idx + i]
            elif q_cur[start_idx + i] >= self.queue_q_list[replace_idxs[i]]:
                # replace while current q value is larger than the element in queue
                self.queue[replace_idxs[i]] = cp.copy(experiences[start_idx + i])
                self.queue_q_list[replace_idxs[i]] = q_cur[start_idx + i]
                self.index_dict[pair_idxs[start_idx + i]] = replace_idxs[i]
                self.index_list[replace_idxs[i]] = pair_idxs[start_idx + i]

    @torch.no_grad()
    def push(self, vl_embeds, q_labels, vl_class, labels, img_idxs, text_all, d_labels=None):
        exps = []
        q_cur = []
        pair_idxs = []
        for k in range(vl_embeds.shape[0]):
            q_label = q_labels[k]
            exps.append(TypicalExperience(
                image_id=img_idxs[k],
                text=text_all[k],
                vl_embedding=to_np(vl_embeds[k]),
                pred_class=to_np(vl_class[k]),
                label=to_np(labels[k]),
                q_value=to_np(q_label)
            ))
            if d_labels is not None:
                d_label = d_labels[k]
                q_cur.append(abs(q_label[0].item() + d_label[0].item()))
            else:
                q_cur.append(abs(q_label[0].item()))
            pair_idxs.append('%s' % exps)
        self._push(experiences=exps, q_cur=q_cur, pair_idxs=pair_idxs)

    def sample(self):
        queue_embeds = []
        queue_labels = []
        queue_q = []

        if self.full:
            up_bound = self.capacity
        else:
            up_bound = self.idx

        # sample_prob = torch.softmax(torch.as_tensor(self.queue_q_list[: up_bound]) + 0.25 * torch.rand(up_bound), dim=0)
        # self._idxs = np.random.choice(
        #     range(up_bound), size=int(self.args.sample_size if self.args.sample_size < up_bound else up_bound),
        #     replace=False, p=sample_prob.detach().numpy())
        self._idxs = np.random.choice(
            range(up_bound), size=int(self.args.sample_size if self.args.sample_size < up_bound else up_bound), replace=False)

        for j in self._idxs:
            queue_embeds.append(torch.as_tensor(self.queue[j].vl_embedding))
            queue_labels.append(torch.as_tensor(self.queue[j].label))
            queue_q.append(torch.as_tensor(self.queue[j].q_value))

        queue_embeds = torch.stack(queue_embeds, dim=0)
        queue_labels = torch.stack(queue_labels, dim=0)
        queue_q = torch.stack(queue_q, dim=0)

        return queue_embeds, queue_labels, queue_q

    def alter_q_value(self, q_value):
        for i in range(len(q_value)):
            self.queue[self._idxs[i]].q_value = q_value[i]

    def load_state_dict(self, state_dict):
        self.args = state_dict['args']
        self.capacity = state_dict['capacity']
        self.queue = state_dict['queue']
        self.queue_q_list = state_dict['queue_q_list']
        self.idx = state_dict['idx']
        self._idxs = state_dict['_idxs']
        self.full = state_dict['full']
        self.empty = state_dict['empty']



