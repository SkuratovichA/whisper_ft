import whisper
from whispermodelmodule import WhisperModelModule, Config
from local_datasets import AlbaizynDataset, load_manifests, WhisperDataCollatorWithPadding
from tqdm import tqdm
import re
import json
import numpy as np
try:
    from safe_gpu import safe_gpu
    gpu_owner = safe_gpu.GPUOwner(1)
    DEVICE = 'gpu'
except:
    DEVICE = 'cpu'

import torch


def transcribe(loader, whisper_model, woptions, wtokenizer):
    def filter_text(text: str):
        text = text.lower()
        text = re.sub(r'[\.,?¿]', '', text)
        return text
    t_wer, t_cer = [], []

    with torch.inference_mode():
        preds = []
        refs = []
        for b in tqdm(loader):
            input_ids = b['input_ids'].half().cuda()
            labels = b['labels'].long().cuda()
            audio_filepaths = b['audio_filepaths']
            results = whisper_model.model.decode(input_ids, woptions)

            for pred, path in zip(results, audio_filepaths):
                preds.append({
                    'audio_filepath': path,
                    'pred_text': filter_text(pred.text),
                })

            for l in labels:
                l[l == -100] = wtokenizer.eot
                ref = wtokenizer.decode(l, skip_special_tokens=True)
                refs.append({'text': ref})

        results = []

        for pred, ref in zip(preds, refs):
            wer = whisper_model.metrics_wer.compute(references=[ref['text'], ], predictions=[pred['pred_text'], ])*100
            cer = whisper_model.metrics_cer.compute(references=[ref['text'], ], predictions=[pred['pred_text'], ])*100
            t_wer.append(wer)
            t_cer.append(cer)
            results.append({**pred, **ref, 'cer': cer, 'wer': wer})

        metrics = {
            'wer': np.array(t_wer),
            'cer': np.array(t_cer)
        }
        return results, metrics


def main():
    test_manifest = load_manifests('/mnt/matylda3/xskura01/workspace/projects/asr_whisper/manifests', filenames={'test': ['test.json']})['test']
    model_path = 'finetuned.pt'
    state_dict = torch.load(model_path)


    cfg = Config()
    whisper_model = WhisperModelModule(cfg)
    whisper_model.model.load_state_dict(state_dict)
    whisper_model.eval()

    woptions = whisper.DecodingOptions(language='es', without_timestamps=True)
    print(f'task: {woptions.task}')
    wtokenizer = whisper.tokenizer.get_tokenizer(True, language="es", task=woptions.task)
    dataset = AlbaizynDataset(manifest=test_manifest, tokenizer=wtokenizer)

    loader = torch.utils.data.DataLoader(dataset, batch_size=8, collate_fn=WhisperDataCollatorWithPadding())

    preds, metrics = transcribe(loader, whisper_model, woptions, wtokenizer)

    print( '\n'.join([f'{n}: mean {arr.mean():.4f} std {arr.std():.4f}' for n, arr in metrics.items()] ))

    with open('results.json', 'w') as of:
        for line in preds:
            of.write(json.dumps(line) + '\n')

if __name__ == "__main__":
    main()
