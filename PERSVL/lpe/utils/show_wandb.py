import wandb
import os
import json
import time


def load_log_file(file) -> dict:
    log_infos = {}
    with open(file, 'r') as f:
        logs = f.read().split('\n')
    for i in range(len(logs)):
        if logs[i] == '':
            continue
        log_dict = json.loads(logs[i])
        for k, v in log_dict.items():
            if k not in log_infos.keys():
                log_infos[k] = []
            log_infos[k].append(v)
    return log_infos


if __name__ == '__main__':
    coeff = 1e-5
    dataset = 'RSITMD'
    q_capacity = 21440
    for beta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        exp = 'lpe_training_without_queue_scratch_blr_0.00008000_beta_{}_'.format(beta) + \
              'q_capacity_{}_prefill_False_exp_True_sim_theta_0.866_bal_coef_{}'.format(q_capacity, coeff)
        output_dir = os.path.join('/data1/amax/hdb/output-5.0.3/{}'.format(dataset), exp)
        log_info_file = load_log_file(os.path.join(output_dir, 'log.txt'))

        exp_name = '{}_lpe_bal_coef_{}'.format(dataset, coeff)
        config = {
            "device": "NVIDIA RTX A6000",
            "experiment": exp_name,
            "blr": 8e-6,
            "batch_size": 16,
            "ngpus": 1,
            "nodes": 1,
        }
        wandb.init(
            project='LPE for image-text retrieval -- evaluation 5.0.3',
            entity='xdu_hdb',
            name=exp_name,
            config=config,
            notes="LPE"
        )

        size = len(log_info_file['val_txt_r1'])
        max_r_mean = 0
        for j in range(size):
            if max_r_mean < log_info_file['val_r_mean'][j]:
                max_r_mean = log_info_file['val_r_mean'][j]

        log = {
            '{}_max_r_mean'.format(dataset): max_r_mean,
            '{}_beta'.format(dataset): beta
        }
        wandb.log(log)




