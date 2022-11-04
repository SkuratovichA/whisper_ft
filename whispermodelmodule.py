from local_datasets import WhisperDataCollatorWithPadding
import torch
from local_datasets import AlbaizynDataset
from transformers import AdamW
import pytorch_lightning as pl
import whisper
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from pytorch_lightning import LightningModule

from wer import WER
from cer import CER

class Config:
    learning_rate = 0.0001
    weight_decay = 1e-3
    adam_epsilon = 1e-8
    warmup_steps = 100
    batch_size = 48
    num_workers = 4
    num_train_epochs = 6
    gradient_accumulation_steps = 4
    sample_rate = 16000
    dev_batch_size = 24


class WhisperModelModule(LightningModule):
    def __init__(
        self,
        cfg:Config,
        model_name='medium',
        lang='es',
        train_dataset=[],
        dev_dataset=[],
    ) -> None:
        super().__init__()
        self.options = whisper.DecodingOptions(language=lang, without_timestamps=True)
        self.model = whisper.load_model(model_name, in_memory=True)
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language=lang, task=self.options.task)

        for p in self.model.encoder.parameters():
            p.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics_wer = WER()
        self.metrics_cer = CER()

        self.cfg = cfg
        self.__train_dataset = train_dataset
        self.__dev_dataset = dev_dataset

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        input_ids = batch['input_ids']
        labels = batch['labels'].long()
        dec_input_ids = batch['dec_input_ids'].long()

        with torch.no_grad():
            audio_features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        self.log('train loss', loss, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_id):
        input_ids = batch['input_ids']
        labels = batch['labels'].long()
        dec_input_ids = batch['dec_input_ids'].long()

        with torch.no_grad():
            audio_features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        out[out == -100] = self.tokenizer.eot
        labels[labels == -100] = self.tokenizer.eot

        o_list, l_list = [], []
        for o, l in zip(out, labels):
            o = torch.argmax(o, dim=1)
            o_list.append(self.tokenizer.decode(o, skip_special_tokens=True))
            l_list.append(self.tokenizer.decode(l, skip_special_tokens=True))
        cer = self.metrics_cer.compute(references=l_list, predictions=o_list) * 100
        wer = self.metrics_wer.compute(references=l_list, predictions=o_list) * 100

        self.log('val/loss', loss, on_step=True, prog_bar=True, logger=True)
        self.log('val/cer', cer, on_step=True, prog_bar=True, logger=True)
        self.log('val/wer', wer, on_step=True, prog_bar=True, logger=True)

        return {
            'cer': cer,
            'wer': wer,
            'loss': loss
        }

    def configure_optimizers(self):
        model = self.model
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params':
                [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay':
                self.cfg.weight_decay
            },
            {'params':
                [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay':
                0.0
            }
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.cfg.learning_rate,
            eps=self.cfg.adam_epsilon
        )
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.warmup_steps,
            num_training_steps=self.t_total
        )
        self.scheduler = scheduler

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.t_total = (
                (len(self.__train_dataset) // (self.cfg.batch_size))
                // self.cfg.gradient_accumulation_steps
                * float(self.cfg.num_train_epochs)
            )

    def train_dataloader(self):
        dataset = AlbaizynDataset(
            manifest=self.__train_dataset,
            tokenizer=self.tokenizer,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=WhisperDataCollatorWithPadding()
        )

    def val_dataloader(self):
        dataset = AlbaizynDataset(
            manifest=self.__dev_dataset,
            tokenizer=self.tokenizer,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.dev_batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=WhisperDataCollatorWithPadding()
        )

