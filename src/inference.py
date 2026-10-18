"""Local artifacts only. No training or Hub downloads in serving."""
import json
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from src.config import LABELS, MAX_LENGTH
from src.tokenization import token_chunks


class SentimentNet(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Linear(encoder.config.hidden_size, len(LABELS))

    def features(self, tokens):
        hidden = self.encoder(**tokens).last_hidden_state
        mask = tokens['attention_mask'].unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)

    def forward(self, tokens):
        return self.head(self.features(tokens))


class Predictor:
    def __init__(self, path, device=None):
        self.meta = json.loads((path / 'artifact.json').read_text())
        if self.meta['labels'] != LABELS:
            raise ValueError('Artifact label order differs from API contract')
        self.version = self.meta['model_version']
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.kind = self.meta['kind']
        if self.kind == 'majority':
            self.majority = self.meta['majority']
            return
        self.tokenizer = AutoTokenizer.from_pretrained(path / 'tokenizer', local_files_only=True)
        self.net = SentimentNet(AutoModel.from_pretrained(path / 'encoder', local_files_only=True))
        self.net.head.load_state_dict(torch.load(path / 'head.pt', map_location='cpu', weights_only=True))
        self.net.to(self.device).eval()

    @torch.inference_mode()
    def predict_proba(self, texts):
        if self.kind == 'majority':
            result = np.zeros((len(texts), 3))
            result[:, self.majority] = 1
            return result
        # Whole articles: non-overlapping token chunks, equal-weight probability mean.
        results = []
        for text in texts:
            chunks = token_chunks(text, self.tokenizer, self.meta['max_length'])
            probs = []
            for start in range(0, len(chunks), 32):
                rows = chunks[start:start+32]
                tokens = self.tokenizer.pad(rows, padding=True, return_tensors='pt').to(self.device)
                probs.append(self.net(tokens).softmax(-1).float().cpu())
            results.append(torch.cat(probs).mean(0).numpy())
        return np.stack(results)


def save_model(path, net, tokenizer, meta):
    path.mkdir(parents=True, exist_ok=True)
    net.encoder.save_pretrained(path / 'encoder', safe_serialization=True)
    tokenizer.save_pretrained(path / 'tokenizer')
    torch.save(net.head.state_dict(), path / 'head.pt')
    (path / 'artifact.json').write_text(json.dumps({**meta, 'labels': LABELS, 'max_length': MAX_LENGTH}, indent=2))
