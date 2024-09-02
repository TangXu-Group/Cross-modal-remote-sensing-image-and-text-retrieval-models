import os
import copy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BaseFinetuning
from vilt.config import ex
from vilt.modules import ViLTransformerSS
from vilt.datamodules.multitask_datamodule import MTDataModule
from torch.optim.optimizer import Optimizer
import warnings
# from model.model import LayerNorm
from torch.nn.modules import LayerNorm, Linear


class MyBackboneFinetuning(BaseFinetuning):

    def __init__(self, unfreeze_backbone_at_epoch: int = 5, train_bn: bool = True, backbone_lr: float = 1e-5):
        super(MyBackboneFinetuning, self).__init__()

    def freeze_clip(self, modules):
        _modules = BaseFinetuning.flatten_modules(modules)
        for mod in _modules:
            self.freeze_module(mod)


    def freeze_module(self, module):
        # if isinstance(module, LayerNorm):
        #     module.track_running_stats = False
        for param in module.parameters(recurse=False):
            param.requires_grad = False

    def freeze_before_training(self, pl_module):
        self.freeze_clip(pl_module.model)

    def finetune_function(
            self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer, opt_idx: int
    ) -> None:
        pass



@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=False)

    model = ViLTransformerSS(_config)
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [MyBackboneFinetuning(), checkpoint_callback, lr_callback]

    # callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )
    print(grad_steps)
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None
    print("this is valvalvla:{}".format(_config["val_check_interval"]))
    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        # devices=1,
        # strategy='ddp',
        num_nodes=_config["num_nodes"],
        # precision=_config["precision"],
        accelerator="ddp",
        # accelerator="gpu",
        # benchmark=True,
        # deterministic=True,
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,
        # max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        # prepare_data_per_node=False,
        # replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        # log_every_n_steps=10,
        # flush_logs_every_n_steps=10,
        num_sanity_val_steps=0,
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",
        # fast_dev_run=_config["fast_dev_run"],
        # auto_lr_find=True,
        # auto_scale_batch_size="power"
        # val_check_interval=1,
        val_check_interval=_config["val_check_interval"]

    )

    if not _config["test_only"]:
        # trainer.test(model, test_dataloaders=dm)
        trainer.fit(model, datamodule=dm)
        # _config["current_epoch"]= "last"
        trainer.test(model, test_dataloaders=dm)
        trainer.test(ckpt_path="best", test_dataloaders=dm)
    else:
        trainer.test(model, datamodule=dm)
