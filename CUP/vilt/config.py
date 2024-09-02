from sacred import Experiment

ex = Experiment("ViLT")


def _loss_names(d):
    ret = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 0,
        "clip":0,
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "vilt"
    seed = 0
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    clip_model= "ViT-L/14"
    current_epoch = 0
    weight_margin = 0.015
    loss_scale_kl=1
    loss_scale_un=1
    loss_scale=1

    # Image setting
    max_image_len = -1
    patch_size = 32
    draw_false_image = 1
    image_only = False

    # Text Setting
    vqav2_label_size = 3129
    max_text_len = 77
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0

    # Transformer Setting
    vit = "vit_base_patch32_384"
    hidden_size = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1
    prompt_length = 16

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-4
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 25000
    warmup_steps = 2500
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads
    clip_zero = True

    # Downstream Setting
    get_recall_metric = False

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False
    # below params varies with the environment
    data_root = ""
    log_dir = "/data/CLIP_wyj/result"
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 16
    linear_projector=0.1


    # Hyperparameter setting

    train_transform_keys = "ViT-L/14"
    val_transform_keys = "ViT-L/14"
    image_size = [14, 14]
    img_features_dim = 1024
    txt_features_dim = 768


    cls_number = 7
    save_image_path = '/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Matchloss-cliploss/images/RSICD'
    json = "/home/amax/wyj/dataset/RSITMD/dataset_RSITMD.json"


# Named configs for "environment" which define gpus and nodes, and paths
@ex.named_config
def env_dandelin():
    data_root = "/data2/dsets/dataset"
    log_dir = "/data2/vilt/result"
    num_gpus = 8
    num_nodes = 1


# Named configs for "task" which define datasets, loss_names and desired batch_size, warmup_steps, epochs, and exp_name
@ex.named_config
def task_mlm_itm():
    exp_name = "mlm_itm"
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096
    max_epoch = 10
    max_image_len = 200


@ex.named_config
def task_mlm_itm_randaug():
    exp_name = "mlm_itm_randaug"
    datasets = ["coco", "vg", "sbu", "gcc"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096
    max_epoch = 10
    max_image_len = 200


@ex.named_config
def task_mlm_itm_mpp():
    exp_name = "mlm_itm_mpp"
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1, "mpp": 1})
    batch_size = 4096
    max_epoch = 10
    max_image_len = 200


@ex.named_config
def task_finetune_nlvr2():
    exp_name = "finetune_nlvr2"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4


@ex.named_config
def task_finetune_nlvr2_randaug():
    exp_name = "finetune_nlvr2_randaug"
    datasets = ["nlvr2"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4


@ex.named_config
def task_finetune_vqa():
    exp_name = "finetune_vqa"
    datasets = ["vqa"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10


@ex.named_config
def task_finetune_vqa_randaug():
    exp_name = "finetune_vqa_randaug"
    datasets = ["vqa"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10


@ex.named_config
def task_finetune_irtr_coco():
    exp_name = "finetune_irtr_coco"
    datasets = ["coco"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4


@ex.named_config
def task_finetune_irtr_coco_randaug():
    exp_name = "finetune_irtr_coco_randaug"
    datasets = ["coco"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4


@ex.named_config
def task_finetune_irtr_f30k():
    exp_name = "finetune_irtr_f30k"
    datasets = ["f30k"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4


@ex.named_config
def task_finetune_irtr_f30k_randaug():
    exp_name = "finetune_irtr_f30k_randaug"
    datasets = ["f30k"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4

@ex.named_config
def task_finetune_irtr_sydney_randaug():
    exp_name = "finetune_irtr_sydney_randaug"
    datasets = ["sydney"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"clip": 1})
    batch_size = 256
    max_epoch = 50
    max_steps = None
    warmup_steps = 0.3
    get_recall_metric = True
    weight_margin = 0.015
    draw_false_text = 0
    learning_rate = 1e-4
    current_epoch = 0
    cls_number = 7
    prompt_length = 16
    clip_zero = True
    save_image_path = "/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Cliploss/images/Sydney"
    json = "/home/amax/wyj/dataset/Sydney_captions/karpathy/sydney.json"
    val_check_interval = 0.1

@ex.named_config
def task_finetune_irtr_nwpu_randaug():
    exp_name = "finetune_irtr_nwpu_randaug"
    datasets = ["nwpu"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"clip": 1})
    batch_size = 256
    max_epoch = 50
    max_steps = None
    warmup_steps = 0.3
    get_recall_metric = True
    weight_margin = 0.015
    draw_false_text = 0
    learning_rate = 1e-4
    current_epoch = 0
    cls_number = 7
    prompt_length = 16
    save_image_path = "/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Cliploss/images/NWPU"
    json = "/home/amax/wyj/dataset/NWPU-Captions-main/karpathy/nwpu_label.json"

@ex.named_config
def task_finetune_irtr_ucm_randaug():
    exp_name = "finetune_irtr_ucm_randaug"
    datasets = ["ucm"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"clip": 1})
    batch_size = 256
    max_epoch = 50
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    weight_margin = 0.015
    draw_false_text = 0
    learning_rate = 1e-4
    current_epoch = 0
    cls_number = 21
    prompt_length = 16
    clip_zero = True
    save_image_path = "/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Cliploss/images/UCM"
    json = "/home/amax/wyj/dataset/UCM_captions/karpathy/ucm.json"




@ex.named_config
def task_finetune_irtr_rsicd_randaug():
    exp_name = "finetune_irtr_rsicd_randaug"
    datasets = ["rsicd"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"clip": 1})
    batch_size = 256
    max_epoch = 30
    max_steps = None
    clip_zero = True
    warmup_steps = 0.1
    get_recall_metric = True
    weight_margin = 0.015
    draw_false_text = 0
    learning_rate = 1e-4
    current_epoch = 0
    cls_number = 30
    prompt_length = 16
    save_image_path = "/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Cliploss/images/RSICD"
    json = "/home/amax/wyj/dataset/RSICD_captions/karpathy/rsicd.json"


@ex.named_config
def task_finetune_irtr_rsitmd_randaug():
    exp_name = "finetune_irtr_rsitmd_randaug"
    datasets = ["rsitmd"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"clip": 1})
    batch_size = 256
    max_epoch = 30
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    weight_margin = 0.015
    draw_false_text = 0
    learning_rate = 1e-4
    current_epoch = 0
    cls_number = 32
    prompt_length = 16
    clip_zero = True
    save_image_path = "/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Cliploss/images/RSITMD"
    json = "/home/amax/wyj/dataset/RSITMD/karpathy/rsitmd.json"
    loss_scale_kl=1
    loss_scale_un=1

##############################ViT-B32###############################################

@ex.named_config
def task_finetune_irtr_ucm_randaug_ViTB32():
    exp_name = "finetune_irtr_ucm_randaug"
    datasets = ["ucm"]
    clip_model = "ViT-B/32"
    train_transform_keys = "ViT-B/32"
    val_transform_keys = "ViT-B/32"
    image_size = [32, 32]
    img_features_dim = 768
    txt_features_dim = 512
    loss_names = _loss_names({"clip": 1})
    batch_size = 256
    max_epoch = 50
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    weight_margin = 0.015
    draw_false_text = 0
    learning_rate = 1e-4
    current_epoch = 0
    cls_number = 21
    prompt_length = 16
    clip_zero = True
    save_image_path = "/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Cliploss/images/UCM"
    json = "/home/amax/wyj/dataset/UCM_captions/karpathy/ucm.json"




@ex.named_config
def task_finetune_irtr_rsicd_randaug_ViTB32():
    exp_name = "finetune_irtr_rsicd_randaug"
    datasets = ["rsicd"]
    clip_model = "ViT-B/32"
    train_transform_keys = "ViT-B/32"
    val_transform_keys = "ViT-B/32"
    image_size = [32, 32]
    img_features_dim = 768
    txt_features_dim = 512
    loss_names = _loss_names({"clip": 1})
    batch_size = 256
    max_epoch = 30
    max_steps = None
    clip_zero = True
    warmup_steps = 0.1
    get_recall_metric = True
    weight_margin = 0.015
    draw_false_text = 0
    learning_rate = 1e-4
    current_epoch = 0
    cls_number = 30
    prompt_length = 16
    save_image_path = "/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Cliploss/images/RSICD"
    json = "/home/amax/wyj/dataset/RSICD_captions/karpathy/rsicd.json"


@ex.named_config
def task_finetune_irtr_rsitmd_randaug_ViTB32():
    exp_name = "finetune_irtr_rsitmd_randaug"
    datasets = ["rsitmd"]
    clip_model = "ViT-B/32"
    train_transform_keys = "ViT-B/32"
    val_transform_keys = "ViT-B/32"
    image_size = [32, 32]
    img_features_dim = 768
    txt_features_dim = 512
    loss_names = _loss_names({"clip": 1})
    batch_size = 256
    max_epoch = 30
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    weight_margin = 0.015
    draw_false_text = 0
    learning_rate = 1e-4
    current_epoch = 0
    cls_number = 32
    prompt_length = 16
    clip_zero = True
    save_image_path = "/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Cliploss/images/RSITMD"
    json = "/home/amax/wyj/dataset/RSITMD/karpathy/rsitmd.json"
    loss_scale_kl=1
    loss_scale_un=1



##############################ViT-B16###############################################

@ex.named_config
def task_finetune_irtr_ucm_randaug_ViTB16():
    exp_name = "finetune_irtr_ucm_randaug"
    datasets = ["ucm"]
    clip_model = "ViT-B/16"
    train_transform_keys = "ViT-B/16"
    val_transform_keys = "ViT-B/16"
    image_size = [16, 16]
    img_features_dim = 768
    txt_features_dim = 512
    loss_names = _loss_names({"clip": 1})
    batch_size = 256
    max_epoch = 50
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    weight_margin = 0.015
    draw_false_text = 0
    learning_rate = 1e-4
    current_epoch = 0
    cls_number = 21
    prompt_length = 16
    clip_zero = True
    save_image_path = "/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Cliploss/images/UCM"
    json = "/home/amax/wyj/dataset/UCM_captions/karpathy/ucm.json"




@ex.named_config
def task_finetune_irtr_rsicd_randaug_ViTB16():
    exp_name = "finetune_irtr_rsicd_randaug"
    datasets = ["rsicd"]
    clip_model = "ViT-B/16"
    train_transform_keys = "ViT-B/16"
    val_transform_keys = "ViT-B/16"
    image_size = [16, 16]
    img_features_dim = 768
    txt_features_dim = 512
    loss_names = _loss_names({"clip": 1})
    batch_size = 256
    max_epoch = 30
    max_steps = None
    clip_zero = True
    warmup_steps = 0.1
    get_recall_metric = True
    weight_margin = 0.015
    draw_false_text = 0
    learning_rate = 1e-4
    current_epoch = 0
    cls_number = 30
    prompt_length = 16
    save_image_path = "/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Cliploss/images/RSICD"
    json = "/home/amax/wyj/dataset/RSICD_captions/karpathy/rsicd.json"


@ex.named_config
def task_finetune_irtr_rsitmd_randaug_ViTB16():
    exp_name = "finetune_irtr_rsitmd_randaug"
    datasets = ["rsitmd"]
    clip_model = "ViT-B/16"
    train_transform_keys = "ViT-B/16"
    val_transform_keys = "ViT-B/16"
    image_size = [16, 16]
    img_features_dim = 768
    txt_features_dim = 512
    loss_names = _loss_names({"clip": 1})
    batch_size = 256
    max_epoch = 30
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    weight_margin = 0.015
    draw_false_text = 0
    learning_rate = 1e-4
    current_epoch = 0
    cls_number = 32
    prompt_length = 16
    clip_zero = True
    save_image_path = "/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Cliploss/images/RSITMD"
    json = "/home/amax/wyj/dataset/RSITMD/karpathy/rsitmd.json"
    loss_scale_kl=1
    loss_scale_un=1



##############################ViT-L14###############################################

@ex.named_config
def task_finetune_irtr_ucm_randaug_ViTL14():
    exp_name = "finetune_irtr_ucm_randaug"
    datasets = ["ucm"]
    train_transform_keys = "ViT-L/14"
    val_transform_keys = "ViT-L/14"
    clip_model = "ViT-L/14"
    image_size = [14, 14]
    img_features_dim = 1024
    txt_features_dim = 768
    loss_names = _loss_names({"clip": 1})
    batch_size = 256
    max_epoch = 50
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    weight_margin = 0.015
    draw_false_text = 0
    learning_rate = 1e-4
    current_epoch = 0
    cls_number = 21
    prompt_length = 16
    clip_zero = True
    save_image_path = "/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Cliploss/images/UCM"
    json = "/home/amax/wyj/dataset/UCM_captions/karpathy/ucm.json"




@ex.named_config
def task_finetune_irtr_rsicd_randaug_ViTL14():
    exp_name = "finetune_irtr_rsicd_randaug"
    datasets = ["rsicd"]
    train_transform_keys = "ViT-L/14"
    val_transform_keys = "ViT-L/14"
    clip_model = "ViT-L/14"
    image_size = [14, 14]
    img_features_dim = 1024
    txt_features_dim = 768
    loss_names = _loss_names({"clip": 1})
    batch_size = 256
    max_epoch = 30
    max_steps = None
    clip_zero = True
    warmup_steps = 0.1
    get_recall_metric = True
    weight_margin = 0.015
    draw_false_text = 0
    learning_rate = 1e-4
    current_epoch = 0
    cls_number = 30
    prompt_length = 16
    save_image_path = "/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Cliploss/images/RSICD"
    json = "/home/amax/wyj/dataset/RSICD_captions/karpathy/rsicd.json"


@ex.named_config
def task_finetune_irtr_rsitmd_randaug_ViTL14():
    exp_name = "finetune_irtr_rsitmd_randaug"
    datasets = ["rsitmd"]
    train_transform_keys = "ViT-L/14"
    val_transform_keys = "ViT-L/14"
    clip_model = "ViT-L/14"
    image_size = [14, 14]
    img_features_dim = 1024
    txt_features_dim = 768
    loss_names = _loss_names({"clip": 1})
    batch_size = 256
    max_epoch = 30
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    weight_margin = 0.015
    draw_false_text = 0
    learning_rate = 1e-4
    current_epoch = 0
    cls_number = 32
    prompt_length = 16
    clip_zero = True
    save_image_path = "/home/amax/wyj/26-CyCLIP-main/CLIP-Retrieval-Cliploss/images/RSITMD"
    json = "/home/amax/wyj/dataset/RSITMD/karpathy/rsitmd.json"




# Named configs for "etc" which are orthogonal to "env" and "task", need to be added at the end


@ex.named_config
def step25k():
    max_epoch = 100
    max_steps = 25000


@ex.named_config
def step50k():
    max_epoch = 100
    max_steps = 50000


@ex.named_config
def step100k():
    max_epoch = 100
    max_steps = 100000


@ex.named_config
def step200k():
    max_epoch = 200
    max_steps = 200000


@ex.named_config
def vit32_base():
    vit = "vit_base_patch32_384"
    patch_size = 32
    hidden_size = 768
    num_heads = 12
    num_layers = 12
