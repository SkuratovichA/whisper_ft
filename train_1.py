from safe_gpu import safe_gpu
gpu_owner = safe_gpu.GPUOwner(1)
DEVICE = 'gpu'

#DEVICE = 'cpu'
import os
import torch
import wandb
import whisper
import evaluate
import torchaudio
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from pathlib import Path
import pytorch_lightning as pl
from whispermodelmodule import WhisperModelModule, Config
from local_datasets import AlbaizynDataset, load_manifests, WhisperDataCollatorWithPadding
from collections import defaultdict
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


os.environ['WANDB_MODE'] = 'offline'


BASE_DIR = Path('/mnt/matylda3/xskura01/workspace/projects/asr_whisper')
MANIFESTS_DIR = BASE_DIR / 'manifests'
BATCH_SIZES = {'train': 32, 'dev': 32, 'test': 32}
TRAIN_RATE = 0.8
AUDIO_MAX_LENGTH = 30.0
TEXT_MAX_LENGTH = 128
SEED = 228
seed_everything(SEED, workers=False)
transcripts_path_list = []  # List of .txt files 


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch=}_{global_step=}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


def setup_training(run_name):
    ex_d = 'experiments'
    if not os.path.exists(ex_d):
        os.makedirs(ex_d)
    run_d = Path(ex_d) / run_name
    id = 0
    while os.path.exists(f'{run_d}_{id}'):
        id += 1
    run_d = f'{run_d}_{id}'
    os.makedirs(run_d)
    return run_d


def main():
    cfg = Config()
    name = 'whisper-es'
    run_directory = setup_training(name)

    filenames = {
        'train': ['train.json', 'cv_train.json', 'cv_dev.json'],
        'dev': ['test.json']
    }
    manifests = load_manifests(MANIFESTS_DIR, filenames)

    train_logger = WandbLogger(
        offline=True,
        name=name,
        project='albaizyn',
        save_dir=BASE_DIR / 'logging',
        entity='SkuratovichA'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(run_directory, 'checkpoints'),
        filename='checkpoint-{epoch:04d}',
        monitor='val/loss',
        save_top_k=2,
    )

    callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval='epoch'), CheckpointEveryNSteps(2000)]
    model_name = 'medium'
    model = WhisperModelModule(cfg, model_name=model_name, lang='es', train_dataset=manifests['train'], dev_dataset=manifests['dev'])

    trainer = Trainer(
        precision=16,
        accelerator=DEVICE,
        gpus=1,
        max_epochs=cfg.num_train_epochs,
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        callbacks=callback_list,
        logger=train_logger,
        log_every_n_steps=50
    )

    trainer.fit(model)
    trainer.save(model.model, 'finetuned.pt')


main()
