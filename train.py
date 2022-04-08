#
#
#      0=================================0
#      |    KPConv in Lightning    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Training Implementation
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Zhening Huang - 30/03/2022
#

import os
import yaml
import pytorch_lightning as pl
from typing import List
from argparse import ArgumentParser
from torch.utils.data import DataLoader, ConcatDataset
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch
from models.KPConv import LightningNetwork, SemanticKittiConfig
import time

config = SemanticKittiConfig()

def prepare_pl_trainer(config: dict) -> pl.Trainer:
    # tensorboard logger
    tb_dir = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
    tb_logger = pl.loggers.TensorBoardLogger(
        tb_dir, default_hp_metric=False
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    # checkpoint callback
    pl_trainer = pl.Trainer(
        replace_sampler_ddp=False,
        logger=tb_logger,
        gpus=config.gpus,
        max_epochs=config.max_epoch,
        default_root_dir=config.output_dir,
        accelerator=config.distributed_backend,
        num_sanity_val_steps=config.num_sanity_val_steps,
        reload_dataloaders_every_n_epochs = 1
    )
    return pl_trainer

def main():
    pl.seed_everything(1234)
    model = LightningNetwork(config)
    pl_trainer = prepare_pl_trainer(config)
    pl_trainer.fit(model)
    # training
    pl_trainer.fit(model)

    # # testing
    # result = pl_trainer.test(model)
    # print(result)


if __name__ == "__main__":
    main()