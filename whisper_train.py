import os
import json
import torch
import random
import string
import argparse
import logging
from pathlib import Path
from infer import transcribe
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from whispermodelmodule import WhisperModelModule, Config
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from local_datasets import AlbaizynDataset, load_manifests, WhisperDataCollatorWithPadding

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '[%(asctime)s - %(funcName)12s() ] >>> %(message)s',
    '%H:%M'
))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def get_name():
    return 'whisper_' + ''.join(random.choice(string.ascii_letters) for _ in range(10))


def get_parser() -> argparse.Namespace:
    """
    Defines the arguments and takes care about parsing

    Returns:
        args: namespace with parsed arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', required=False, type=str, default=get_name(), help='Name of an experiment')

    parser.add_argument('--base_dir', required=True, type=str, help='Base working directory')
    parser.add_argument(
        '--manifest_dir',
        required=False,
        type=str, default=None,
        help='Directory where dataset manifest are stored. `base_dir`/manifests if not specified'
    )

    parser.add_argument('--train_batch_size', type=int, required=False, default=16)
    parser.add_argument('--dev_batch_size', type=int, required=False, default=32)
    parser.add_argument('--test_batch_size', type=int, required=False, default=32)

    parser.add_argument('--learning_rate', type=float, required=False, default=0.00001)
    parser.add_argument('--weight_decay', required=False, type=float, default=1e-3)
    parser.add_argument('--adam_epsilon', required=False, type=float, default=1e-8)
    parser.add_argument('--num_workers', required=False, type=int, defualt=4)
    parser.add_argument('--num_train_epochs', required=False, type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', required=False, default=1, type=int)
    parser.add_argument('--precision', required=False, default=16, type=int, options=[16, 32])

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--test_results_json_filepath', required=False, type=str, default='results.json')
    parser.add_argument(
        '--test_manifest_with_references',
        action='store_true',
        help="'text' field with reference if in the test manifest"
    )

    parser.add_argument('--train_manifests', required=True, nargs='+', help='Names of train manifests in NeMo format')
    parser.add_argument('--dev_manifests', required=True, nargs='+', help='Names of dev manifests in NeMo format')
    parser.add_argument('--test_manifests', required=False, nargs='+', help='Names of test manifests in NeMo format')

    parser.add_argument('--seed', type=int, required=False, default=42)

    parser.add_argument('--save_path', required=False, default=None, type=str, help='Path where to store .pt model')
    parser.add_argument('--log_every_n_steps', required=False, type=int, default=50)
    parser.add_argument('--checkpoint_every_n_steps', required=False, type=int, default=2000)
    parser.add_argument('--save_top_k', required=False, default=2)

    parser.add_argument('--lang', required=True, type=str)
    parser.add_argument('--model_name', required=False, default='medium',
                        options=['tiny', 'medium', 'small', 'base', 'large'])

    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_entity', required=False, type=str)
    parser.add_argument('--wandb_project', required=False, type=str)

    parser.add_argument('--experiment_directory', required=False, type=str, default='experiments')

    parser.add_argument('--online', action='store_false')
    parser.add_argument('--gpus', required=False, default=1, type=int)

    args = parser.parse_args()

    if not args.manifest_dir:
        args.manifest_dir = Path(args.base_dir) / 'manifests'
    if not args.save_path:
        args.save_path = Path(args.base_dir) / f'{args.name}_model.pt'
    if not args.do_test and args.test_results_json_filepath:
        logger.warning('--do_test is False, but test_results_json_filepath specified. Have you forgotten something?')
    if not args.do_test and not args.do_train:
        logger.warning('Suspicious... --do_test nor --do_train have not been specified.')

    return args


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


def main():
    args = get_parser()

    from safe_gpu import safe_gpu
    gpu_owner = safe_gpu.GPUOwner(args.gpus)
    seed_everything(args.seed, workers=False)
    if not args.online:
        os.environ['WANDB_MODE'] = 'offline'

    # set up training
    train_logger = (
        WandbLogger(
            offline=not args.online,
            name=args.name,
            project=args.wandb_project,
            save_dir=args.base_dir / 'logging',
            entity=args.wandb_entity,
        )
        if args.wandb
        else None
    )

    cfg = Config(**{k: v for k, v in list(args.vars().items()) if k in list(Config.__annotations__.keys())})

    manifest_filenames = {
        'train': args.train_manifests,
        'dev': args.dev_manifests,
        'test': args.test_manifests,
    }
    manifests = load_manifests(args.manifest_dir, manifest_filenames)

    callback_list = [
        ModelCheckpoint(
            dirpath=os.path.join(Path(args.experiment_directory) / args.name, 'checkpoints'),
            filename='checkpoint-{epoch:04d}',
            monitor='val/loss',
            save_top_k=args.save_top_k,
        ),
        LearningRateMonitor(logging_interval='epoch'),
        CheckpointEveryNSteps(args.checkpoint_every_n_steps)
    ]
    model = WhisperModelModule(
        cfg=cfg,
        model_name=args.model_name,
        lang=args.lang,
        train_dataset=manifests['train'],
        dev_dataset=manifests['dev']
    )
    trainer = Trainer(
        precision=args.precision,
        accelerator='gpu',
        gpus=args.gpus,
        max_epochs=cfg.num_train_epochs,
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        callbacks=callback_list,
        logger=train_logger,
        log_every_n_steps=args.log_every_n_steps
    )

    if args.do_train:
        trainer.fit(model)
    trainer.save(model.model, args.save_path)

    if args.do_test:
        logger.info('Start testing...')
        model.eval()
        woptions = whisper.DecodingOptions(language=args.lang, without_timestamps=True)
        wtokenizer = whisper.tokenizer.get_tokenizer(True, language=args.lang, task=woptions.task)
        dataset = AlbaizynDataset(manifest=manifests['test'], tokenizer=wtokenizer)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.test_batch_size,
            collate_fn=WhisperDataCollatorWithPadding()
        )
        preds, metrics = transcribe(
            loader,
            model,
            woptions,
            wtokenizer,
            with_references=args.test_manifest_with_references
        )

        if metrics:
            logger.info(
                'Testing done:\n' +
                '\n'.join([f'{n}: mean {arr.mean():.4f} std {arr.std():.4f}' for n, arr in metrics.items()])
            )

        logger.info(f'Writing results to {args.test_results_json_filepath}')
        with open(args.test_results_json_filepath, 'w') as of:
            for line in preds:
                of.write(json.dumps(line) + '\n')


main()
