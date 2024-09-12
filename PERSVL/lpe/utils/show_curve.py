import matplotlib.pyplot as plt
import json
import os


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
            log_infos[k].append(float(v))
    return log_infos


def cal_avg(datas, beta=0.99):
    avg_list = [datas[0]]
    avg = datas[0]
    for i in range(1, len(datas)):
        avg = beta*avg + (1 - beta)*datas[i]
        avg_list.append(avg)
    return avg_list


def show_curve(name, datas, expavg=None):
    plt.figure()
    plt.title(name)

    x_coordinate = range(len(datas))
    if expavg is None:
        plt.plot(x_coordinate, datas)
    else:
        plt.plot(x_coordinate, datas, c='b')
        plt.plot(x_coordinate, expavg, c='r')
    plt.xlabel('step')
    plt.ylabel(name)
    plt.show()


def main(file):
    log_infos = load_log_file(file)
    for k, v in log_infos.items():
        avg_list = None
        if 'lr' not in k:
            avg_list = cal_avg(v[:])
        show_curve(k[:], v[:], avg_list[:])


if __name__ == '__main__':
    # experiment = 'lpe_training_without_queue_only_discriminator_loss_blr_0.00000800_set_baseline_True_reward_baseline_0.5_r_coeff_0.8_r_mode_2_q_capacity1024_sample_rate_0.5'
    # path = r'/data1/amax/hdb/output/'
    # path = os.path.join(path, experiment)
    path = r'/data1/amax/hdb/output-5.1.1/RSITMD/CRSRT_from_scratch_blr_0.00004000_low_level_stage_2_use_visual_cls_False'
    main(os.path.join(path, 'log_info.txt'))


