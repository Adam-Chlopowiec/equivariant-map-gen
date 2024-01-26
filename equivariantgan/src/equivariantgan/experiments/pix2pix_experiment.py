import os
from logging import getLogger

import hydra
import torch
import pytorch_lightning as pl
from copy import deepcopy
from omegaconf import DictConfig


torch.set_float32_matmul_precision('highest')
# torch.backends.cudnn.benchmark = True
logger = getLogger(__name__)


@hydra.main(config_path=os.environ["CONFIG_DIR"], config_name="default")
def run_experiment(config: DictConfig) -> float:
    """Basic experiment.

    Args:
        config (DictConfig): Hydra config (see Hydra documentation).

    Returns:
        float: Main metric. This is ONLY for Optuna. To change it to multi-objective or something else, see Optuna docs.
    """
    # initialize datamodule
    logger.info(f"Instantiate <{config.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(config.datamodule, _recursive_=False)

    datamodule.setup()
    datamodule.prepare_data()

    # initialize model
    logger.info(f"Instantiate <{config.lightning_model._target_}>")
    model = hydra.utils.instantiate(
        config.lightning_model,
        _recursive_=False
    )

    # initalize logger
    logger.info(f"Instantiate <{config.logger._target_}>")
    pl_logger = hydra.utils.instantiate(config.logger)
    pl_logger.config = config

    callbacks = build_callbacks(config.callbacks)

    logger.info("Instantiate <Trainer>")
    trainer = pl.Trainer(
        logger=pl_logger, callbacks=callbacks, **config.training.trainer
    )
    # train_loader = datamodule.train_dataloader()
    # trainer.fit(model, train_dataloaders=[datamodule.train_dataloader()], val_dataloaders=[datamodule.test_dataloader()])
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    run_experiment()