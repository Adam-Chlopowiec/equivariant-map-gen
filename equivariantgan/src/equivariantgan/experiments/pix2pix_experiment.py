import os
import argparse
import yaml
from logging import getLogger

import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from copy import deepcopy
from omegaconf import DictConfig, OmegaConf

from equivariantgan.datamodule.map_datamodule import MapDataModule
from equivariantgan.dataset.map_dataset import MapDataset
from equivariantgan.training.map_gan import MapGan


torch.set_float32_matmul_precision('highest')
# torch.backends.cudnn.benchmark = True
logger = getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='Path to config')


def run_experiment():
    opt, _ = parser.parse_args()
    with open(opt.config, 'r') as f:
        config = OmegaConf.load(f)
    
    # initialize datamodule
    logger.info(f"Instantiate Datamodule")
    datamodule = MapDataModule(
        data_root=config.datamodule.data_root,
        target_size=config.datamodule.target_size,
        num_workers=config.datamodule.num_workers,
        batch_size=config.datamodule.batch_size
    )

    datamodule.setup()
    datamodule.prepare_data()

    # initialize model
    logger.info(f"Instantiate <{config.lightning_model._target_}>")
    model = MapGan(**config.lightning_model)

    # initalize logger
    logger.info(f"Instantiate <{config.logger._target_}>")
    pl_logger = WandbLogger(**config.logger)
    pl_logger.config = config

    callbacks = [
        pl.callbacks.ModelCheckpoint(**config.callbacks.model_checkpoint),
    ]

    logger.info("Instantiate <Trainer>")
    trainer = pl.Trainer(
        logger=pl_logger, callbacks=callbacks, **config.training.trainer
    )
    # train_loader = datamodule.train_dataloader()
    # trainer.fit(model, train_dataloaders=[datamodule.train_dataloader()], val_dataloaders=[datamodule.test_dataloader()])
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    run_experiment()