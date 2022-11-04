from safe_gpu import safe_gpu
gpu_owner = safe_gpu.GPUOwner(1)
import inspect


"""
Fine-tuning OpenAI Whisper models for speech recognition.
"""
from data_classes import ModelArguments, DataTrainingArguments, WhisperDataCollatorWithPadding

from cer import CER
from wer import WER
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
# flake8: noqa: E501
import logging
import os
import re

import torchaudio
import whisper
import sys

import numpy as np
import torch

import datasets
from datasets import DatasetDict, load_dataset
import transformers
from torch import nn
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
    Seq2SeqTrainer,
    WhisperModel,
    WhisperConfig,
    Trainer
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import wandb


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0.dev0")

require_version("datasets>=1.18.0", "To fix: pip install -r examples/pytorch/speech-recognition/requirements.txt")

logger = logging.getLogger(__name__)



def write_wandb_pred(pred_str, label_str, prefix="eval"):
    # convert str data to a wandb compatible format
    str_data = [[label_str[i], pred_str[i]] for i in range(len(pred_str))]
    # we'll log all predictions for the last epoch
    wandb.log(
        {
            f"{prefix}/predictions": wandb.Table(
                columns=["label_str", "pred_str"], data=str_data
            )
        },
    )



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.info(training_args)
    # Set wandb project ID before instantiating the Trainer
    os.environ["WANDB_PROJECT"] = data_args.wandb_project
    os.environ["WANDB_MODE"] = 'offline'
    report_to_wandb = "wandb" in training_args.report_to

    sample_rate = 16000

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s - %(name)s] >>> %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # load the model 
    logger.info(f"model path: {model_args.model_name_or_path}")
    model = whisper.load_model(model_args.model_name_or_path, in_memory=True)
    # model = WhisperModel.from_pretrained(model_args.model_name_or_path)

    # set the dropout for the MLP layers -> we do this here as the MLP layers are written as a 'sequential'
    # so changing the modelling code gives mis-matches in the state-dict

    if not model_args.freeze_encoder:
        # only apply dropout when training the encoder
        for block_idx in range(len(model.encoder.blocks)):
            mlp_layer = model.encoder.blocks[block_idx].mlp
            # going very verbose to explain what we're doing here!
            fc1 = mlp_layer[0]
            act_fn = mlp_layer[1]
            dropout = nn.Dropout(p=model_args.dropout_rate)
            fc2 = mlp_layer[2]
            model.encoder.blocks[block_idx].mlp = nn.Sequential(fc1, act_fn, dropout, fc2, dropout)

    """for block_idx in range(len(model.decoder.blocks)):
        mlp_layer = model.decoder.blocks[block_idx].mlp
        fc1 = mlp_layer[0]
        act_fn = mlp_layer[1]
        dropout = nn.Dropout(p=model_args.dropout_rate)
        fc2 = mlp_layer[2]
        model.decoder.blocks[block_idx].mlp = nn.Sequential(fc1, act_fn, dropout, fc2, dropout)"""

    # load the tokenizer
    whisper_tok = whisper.tokenizer.get_tokenizer(False, task="transcribe", language="es")
    tokenizer = whisper_tok.tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    # 4. Load dataset

    # raw_datasets = load_dataset(
    #     "csv",
    #     data_files={
    #         i: f'datasets/{i}_metadata_fullpath.csv' for i in ["train", "dev", "test"]
    #     }
    # )

    raw_datasets = load_dataset(
        'audiofolder',
        data_dir=data_args.dataset_name_or_path,
        streaming=True
    )
    logger.info(raw_datasets)


    if not training_args.do_train and not training_args.do_eval and not training_args.do_predict:
        raise ValueError(
            "Cannot not train, not do evaluation and not do prediction. At least one of "
            "training, evaluation or prediction has to be done."
        )

    # if not training, there is no need to run multiple epochs
    if not training_args.do_train:
        training_args.num_train_epochs = 1

    # 6.
    # raw_datasets = raw_datasets.cast_column(
    #     data_args.audio_column_name, datasets.features.Audio(sampling_rate=sample_rate)
    # )

    # 7. Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = int(data_args.max_duration_in_seconds * sample_rate)
    min_input_length = min(int(data_args.min_duration_in_seconds * sample_rate), 1)
    max_eval_input_length = int(data_args.max_eval_duration_in_seconds * sample_rate) if data_args.max_eval_duration_in_seconds else None
    max_target_length = data_args.max_target_length
    min_target_length = data_args.min_target_length
    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers
    text_column_name = data_args.text_column_name
    do_lower_case = data_args.do_lower_case
    dataset_name_or_path = data_args.dataset_name_or_path

    def prepare_dataset(batch):
        # 'Error correction' of targets
        input_str = batch[text_column_name]

        # Finally, we tokenize the processed text
        input_ids_column = 'mel'  # 'input_features'
        transcript_column = 'tokens'  # decoder_input_ids
        # batch["input_features"] = batch[audio_column_name]['array']
        batch[input_ids_column] = batch[audio_column_name]['array']
        del batch[audio_column_name]
        del batch[text_column_name]
        batch["input_lengths"] = len(batch[input_ids_column])
        batch[transcript_column] = tokenizer(input_str).input_ids
        return batch

    raw_datasets = raw_datasets.with_format('torch')
    vectorized_datasets = raw_datasets.map(
        prepare_dataset
    )

    # filter training data with inputs longer than max_input_length
    def is_audio_in_length_range(input_length):
        return min_input_length < input_length < max_input_length

    if training_args.do_train:
        vectorized_datasets["train"] = vectorized_datasets["train"].filter(
            is_audio_in_length_range,
            input_columns=["input_lengths"],
        )


    # filter training data with targets shorter than min_target_length or longer than max_target_length
    def is_labels_in_length_range(labels):
        return min_target_length < len(labels) < max_target_length


    # for large datasets it is advised to run the preprocessing on a
    # single machine first with `args.preprocessing_only` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step `args.preprocessing_only` can then be set to `False` to load the
    # cached dataset
    if data_args.preprocessing_only:
        cache = {k: v.cache_files for k, v in vectorized_datasets.items()}
        logger.info(f"Data preprocessing finished. Files cached at {cache}.")
        return

    if model_args.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
        # model.freeze_encoder()
        logging.info("Model encoder has been frozen")

    # 8. Load Metric
    metric_wer = WER  # datasets.load_metric("wer")
    metric_cer = CER  # datasets.load_metric("cer")


    def compute_metrics(pred):
        pred_ids = pred.predictions
        pred.label_ids[pred.label_ids == -100] = tokenizer.eos_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        pred_str = [x.lstrip().strip() for x in pred_str]

        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)

        wer = metric_wer.compute(predictions=pred_str, references=label_str)
        cer = metric_cer.compute(predictions=pred_str, references=label_str)

        #return {"wer": wer, "cer": cer, "wer_norm": wer_norm, "cer_norm": cer_norm}
        return {"wer": wer, "cer": cer}

    def compute_metrics_and_predictions(pred):
        pred_ids = pred.predictions
        pred.label_ids[pred.label_ids == -100] = tokenizer.eos_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        pred_str = [x.lstrip().strip() for x in pred_str]

        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)

        wer = metric_wer.compute(predictions=pred_str, references=label_str)
        cer = metric_cer.compute(predictions=pred_str, references=label_str)

        return {"wer": wer, "cer": cer, "pred_str": pred_str, "label_str": label_str}

    # Define data collator
    eos = tokenizer.eos_token_id
    t_stamp = tokenizer("<|notimestamps|>").input_ids[0]
    whisper_data_collator = WhisperDataCollatorWithPadding(eos_token_id=eos, time_stamp_token_id=t_stamp)

    # make sure model uses 50257 as BOS
    # bos = tokenizer("<|startoftranscript|>").input_ids[0]
    # model.config.decoder_start_token_id = bos

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=vectorized_datasets['train'] if training_args.do_train else None,
        eval_dataset=vectorized_datasets['validation'] if training_args.do_eval else None,
        data_collator=whisper_data_collator,
    )
    print('inspected signature columns: ', inspect.signature(model.forward))
    print('data collator', trainer.data_collator)
    print('signature columns', trainer._signature_columns)

    # 8. Finally, we can start training

    # Training
    if training_args.do_train:

        # use last checkpoint if exist
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(output_dir='finetuned_models')

        metrics = train_result.metrics

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()

    # Change decoding strategy for final eval/predict
    if training_args.do_eval or training_args.do_predict:
       trainer.model.num_beams = 2

    trainer.compute_metrics = compute_metrics_and_predictions

    results = {}
    if training_args.do_eval:
        if not training_args.do_train and report_to_wandb:
            # manually init wandb
            wandb.init(project=data_args.wandb_project, name=training_args.run_name)
        # Have to run this as a predict step, otherwise trainer will try to log the pred/label strings to wandb
        eval_results = trainer.predict(
            vectorized_datasets["test"],
            num_beams=model_args.num_beams,
            length_penalty=model_args.length_penalty
        )
        metrics = eval_results.metrics
        pred_str = metrics.pop("eval_pred_str", None)
        label_str = metrics.pop("eval_label_str", None)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        if report_to_wandb:
            metrics = {os.path.join("eval", k[len("eval") + 1:]): v for k, v in metrics.items()}
            wandb.log(metrics)
            write_wandb_pred(pred_str, label_str, prefix="eval")

    # if training_args.do_predict:
    #     if not training_args.do_train and not training_args.do_eval and report_to_wandb:
    #         # manually init wandb
    #         wandb.init(project=data_args.wandb_project, name=training_args.run_name)
    #     for split in test_split:
    #         predict_results = trainer.predict(
    #             vectorized_datasets[split],
    #             metric_key_prefix=split,
    #             num_beams=model_args.num_beams,
    #             length_penalty=model_args.length_penalty
    #         )
    #         metrics = predict_results.metrics
    #         pred_str = metrics.pop(f"{split}_pred_str", None)
    #         label_str = metrics.pop(f"{split}_label_str", None)

    #         trainer.log_metrics(split, metrics)
    #         trainer.save_metrics(split, metrics)

    #         if report_to_wandb:
    #             metrics = {os.path.join(split, k[len(split)+1:]): v for k, v in metrics.items()}
    #             wandb.log(metrics)
    #             write_wandb_pred(pred_str, label_str, prefix=split)

    return results


if __name__ == "__main__":
    main()
