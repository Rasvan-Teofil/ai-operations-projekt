"""Fixed training protocol. Never reads val or test texts/labels."""
import argparse
import json
import random
import time
import numpy as np
import pandas as pd
import torch
import mlflow
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from src.config import *
from src.inference import SentimentNet, save_model
from src.tracking import tracking, verify_data, fingerprint


def seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)


def main(kind):
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required: refusing CPU training')
    seed()
    manifest = verify_data()
    path = MODELS / kind
    if path.exists() or (REPORTS / 'evaluation.lock').exists():
        raise SystemExit('Artifact already exists or test evaluation started. Refusing retraining.')
    tracking()
    df = pd.read_csv(DATA / 'train.csv')
    y = np.array([LABELS.index(v) for v in df.label])
    texts = df.text.tolist()
    with mlflow.start_run(run_name=kind) as run:
        params = {'kind': kind, 'seed': SEED, 'train_rows': len(df), 'encoder': ENCODER,
                  'encoder_revision': ENCODER_REVISION, 'max_length': MAX_LENGTH,
                  'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
                  'cuda_runtime': torch.version.cuda, 'test_sha256': manifest['splits']['test']['sha256'],
                  'val_policy': 'never read', 'selection': 'fixed final epoch; test once at end'}
        meta = {'kind': kind, 'model_version': f'{kind}-{SEED}-v1', 'run_id': run.info.run_id}
        if kind == 'majority':
            path.mkdir(parents=True)
            meta.update(labels=LABELS, majority=int(np.bincount(y, minlength=3).argmax()))
            (path / 'artifact.json').write_text(json.dumps(meta, indent=2))
        else:
            if kind in ('frozen', 'centroid'):
                tokenizer = AutoTokenizer.from_pretrained(ENCODER, revision=ENCODER_REVISION)
                net = SentimentNet(AutoModel.from_pretrained(ENCODER, revision=ENCODER_REVISION)).cuda()
            else:
                from src.inference import Predictor
                predictor = Predictor(MODELS / 'frozen', device='cuda')
                net, tokenizer = predictor.net, predictor.tokenizer
            tokens = tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding=False)
            def batch(indices):
                rows = [{k: v[int(i)] for k,v in tokens.items()} for i in indices]
                return tokenizer.pad(rows, padding=True, return_tensors='pt').to('cuda')
            started = time.perf_counter()
            if kind in ('frozen', 'centroid'):
                net.eval()
                vectors = []
                with torch.inference_mode():
                    for start in range(0, len(df), 64):
                        vectors.append(net.features(batch(np.arange(start, min(start+64,len(df))))).cpu().numpy())
                features = np.concatenate(vectors)
                if kind == 'frozen':
                    classifier = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=SEED)
                    classifier.fit(features, y)
                    weight, bias = classifier.coef_, classifier.intercept_
                    params.update(C=1., class_weight='balanced', head='logistic regression')
                else:
                    centers = np.stack([features[y == i].mean(0) for i in range(3)])
                    weight, bias = 2*centers, -(centers**2).sum(1)
                    params.update(head='nearest class centroid', distance='squared Euclidean')
                with torch.no_grad():
                    net.head.weight.copy_(torch.tensor(weight, dtype=torch.float32, device='cuda'))
                    net.head.bias.copy_(torch.tensor(bias, dtype=torch.float32, device='cuda'))
                params.update(encoder_frozen=True, batch_size=64)
            else:
                # One predefined stronger model: all encoder layers updated, full train split.
                net.train()
                optimizer = torch.optim.AdamW([
                    {'params': net.encoder.parameters(), 'lr': 2e-5},
                    {'params': net.head.parameters(), 'lr': 2e-4}], weight_decay=.01)
                batch_size, epochs = 32, 3
                steps = epochs * int(np.ceil(len(df)/batch_size))
                schedule = get_linear_schedule_with_warmup(optimizer, int(.1*steps), steps)
                weights = torch.tensor(len(y)/(3*np.bincount(y)), dtype=torch.float32, device='cuda')
                criterion = torch.nn.CrossEntropyLoss(weight=weights)
                for epoch in range(epochs):
                    order = np.random.default_rng(SEED+epoch).permutation(len(df))
                    total, seen = 0., 0
                    for start in range(0, len(df), batch_size):
                        ix = order[start:start+batch_size]
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            logits = net(batch(ix))
                            loss = criterion(logits.float(), torch.tensor(y[ix], device='cuda'))
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
                        optimizer.step()
                        schedule.step()
                        total += loss.item()*len(ix)
                        seen += len(ix)
                        if start % (batch_size*50) == 0:
                            print(f'epoch={epoch+1}/3 rows={seen}/{len(df)} loss={total/seen:.4f}', flush=True)
                    mlflow.log_metric('train_loss', total/seen, step=epoch+1)
                    mlflow.log_metric('train_rows_seen', seen, step=epoch+1)
                params.update(encoder_frozen=False, epochs=epochs, batch_size=batch_size, encoder_lr=2e-5,
                              head_lr=2e-4, weight_decay=.01, precision='bf16', initialization='frozen head',
                              checkpoint_rule='final epoch 3, no validation selection')
            torch.cuda.synchronize()
            mlflow.log_metric('training_seconds', time.perf_counter()-started)
            save_model(path, net, tokenizer, meta)
        mlflow.log_params(params)
        mlflow.log_dict(manifest, 'data_manifest.json')
        digest = fingerprint(path)
        mlflow.log_param('artifact_sha256', digest)
        mlflow.log_artifact(str(path / 'artifact.json'), 'model_metadata')
        print(f'{kind} ready: {run.info.run_id}, artifact {digest}', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['frozen', 'finetuned', 'centroid', 'majority'])
    main(parser.parse_args().kind)
