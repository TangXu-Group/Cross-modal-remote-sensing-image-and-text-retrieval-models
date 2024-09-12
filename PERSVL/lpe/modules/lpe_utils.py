import numpy as np
from torch import nn
import torch


class Tmp_input:
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def get_probs(actions, probs):
    prob_list = None
    for i in range(actions.shape[0]):
        if prob_list is None:
            prob_list = probs[i, actions[i]].unsqueeze(0)
        else:
            prob_list = torch.cat([prob_list, probs[i, actions[i]].unsqueeze(0)], dim=0)
    return prob_list


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def momentum_update_params(net, target_net, tau):
    """momentum update current network"""
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def eta_sched(epoch, max_epoch):
    assert epoch <= max_epoch
    x = epoch / max_epoch
    return -(x**3) + 1


def alter_idx(args, text, idx, txt2id_dict, txt2img):
    idx_t = idx.contiguous().view(-1, 1)
    mask = torch.eq(idx_t, idx_t.T)

    # for i in range(len(idx)):
    #     for j in range(len(text)):
    #         if idx[i] in txt2img[txt2id_dict[text[j]]]:
    #             mask[i, j] = True

    # if args.data_name == 'UCM' or args.data_name == 'Sydney':
    # for i in range(len(idx)):
    #     for j in range(len(text)):
    #         if i == j:
    #             continue
    #         if txt2img[txt2id_dict[text[j]]].get(int(idx[i]), False):
    #             mask[i, j] = True

    return mask


@torch.no_grad()
def concat_all_gather(tensor, args):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if args.distributed:
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output
    else:
        return tensor