import os
import copy
import torch
import numpy as np
from pathlib import Path
from torch import utils
from config import ex, config
import pytorch_lightning as pl
from datetime import datetime
from Model import VideoFrameIdenetity
from Dataset import AnimalKingdomDataset
torch.manual_seed(0)

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    datestime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_version = _config['version'] if _config['version'] is not None else datestime_str
    _config['model_dir'] = os.path.join(_config["model_dir"], _config["name"], model_version)
    Path(_config['model_dir']).mkdir(parents=True, exist_ok=True)

    pl.seed_everything(_config["seed"])
    dataset_train = AnimalKingdomDataset(_config, split="train")
    dataset_valid = AnimalKingdomDataset(_config, split="val")
    _config['max_steps'] = _config['max_epochs'] * len(dataset_train) // _config['batch_size']

    model = VideoFrameIdenetity(_config).to(_config['device'])
    model.set_loss_func(_config['loss'])

    train_loader = utils.data.DataLoader(dataset_train, batch_size=_config['batch_size'], shuffle=True, num_workers=_config["data_workers"]) # bugs on MACOS
    valid_loader = utils.data.DataLoader(dataset_valid, batch_size=_config['batch_size'], shuffle=False, num_workers=_config["data_workers"]) # bugs on MACOS

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=_config['model_dir'], 
        filename='{epoch}-{valid_BinaryAccuracy:.3f}-{valid_BinaryPrecision:.3f}-{valid_BinaryRecall:.3f}',
        verbose=True,
        save_top_k=3, 
        every_n_epochs=1,
        monitor="valid_BinaryAccuracy", 
        mode="max", 
        save_last=True)
    summary_callback = pl.callbacks.ModelSummary(max_depth=1)
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")

    csv_logger = pl.loggers.CSVLogger(save_dir=_config["log_dir"], name=_config['name'], version=datestime_str)
    csv_logger.log_hyperparams(_config)
    wandb_logger = pl.loggers.WandbLogger(project='VideoFrameIdentityNetwork', save_dir=_config["log_dir"], name=_config['name'], version=model_version)
    wandb_logger.experiment.config.update(_config, allow_val_change=True)
    trainer = pl.Trainer(max_epochs=_config['max_epochs'], 
                        logger=[csv_logger, wandb_logger], 
                        # log_every_n_steps=50, # 
                        limit_train_batches=_config['limit_train_batches'],
                        limit_valid_batches=_config['limit_val_batches'],
                        log_every_n_steps=(len(dataset_train) // _config['batch_size']) // 3,
                        callbacks=[checkpoint_callback, lr_callback, summary_callback])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader, ckpt_path=_config['ckpt_path'])

