from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
MODELS = ROOT / 'models'
REPORTS = ROOT / 'reports'
LABELS = ['negative', 'neutral', 'positive']
SEED = 20261018
ENCODER = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
ENCODER_REVISION = 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42'
DATASET = 'Kenpache/multilingual-financial-sentiment'
DATA_REVISION = '9d43950d151602f623f28bba62f30554b5d40dd2'
MAX_LENGTH = 128
EXPERIMENT = 'article-sentiment-second-path'
