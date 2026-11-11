"""Pure selection and provenance helpers for future next-stage runs.

Importing this module never starts a run or changes the historical experiment.
"""
import hashlib
import json
import math
from typing import Literal

from pydantic import Field

from src.next_stage import Record, SHA

EXPERIMENT = 'article-sentiment-second-path'


class RunIdentity(Record):
    research_question: str = Field(min_length=1)
    stage: Literal['development', 'final-test', 'engineering']
    data_version: str = Field(min_length=1)
    annotation_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    model_family: str = Field(min_length=1)
    adaptation: str = Field(min_length=1)
    fold: str = Field(min_length=1, description='Use full-train for a final artifact')
    seed: int
    precision: str = Field(min_length=1)
    context_policy: str = Field(min_length=1)
    git_commit: str = Field(pattern=r'^[0-9a-f]{40}$')
    protocol_sha256: str = Field(pattern=SHA)
    input_sha256: str = Field(pattern=SHA)
    annotation_sha256: str = Field(pattern=SHA)
    split_sha256: str = Field(pattern=SHA)
    artifact_sha256: str = Field(pattern=SHA)
    base_model_sha256: str = Field(pattern=SHA)

    def tags(self):
        # Candidate-specific artifacts must NOT enter the comparison-group key.
        shared = {key: getattr(self, key) for key in (
            'stage', 'protocol_sha256', 'input_sha256', 'annotation_sha256', 'split_sha256')}
        group = hashlib.sha256(json.dumps(shared, sort_keys=True).encode()).hexdigest()
        return {**{key: str(value) for key, value in self.model_dump().items()},
                'phase': 'article-next-stage', 'comparable_group': group,
                'label_status': 'AI-labelled · Codex-reviewed', 'human_gold': 'false'}


def primary_score(per_language):
    """Equal DE/EN weight; each value must already be macro-F1 over 3 classes."""
    values = [float(per_language[lang]) for lang in ('de', 'en')]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError('Finite DE/EN macro-F1 in [0,1] required')
    return sum(values) / 2


def select_candidate(records):
    """Within .005 of the best score, warm latency then full standalone bytes.

    Never call this across different comparison groups or missing measurements.
    """
    if not records or len({r['comparable_group'] for r in records}) != 1:
        raise ValueError('A single nonempty comparison group is required')
    if not records[0]['comparable_group'] or len({r['model'] for r in records}) != len(records):
        raise ValueError('Unique candidates and a comparison group are required')
    for row in records:
        for key in ('primary_score', 'inference_ms_per_text', 'full_model_size_bytes'):
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError('Finite nonnegative measurements required')
        if row['primary_score'] > 1 or row['full_model_size_bytes'] <= 0:
            raise ValueError('Invalid F1 or full model size')
    best = max(r['primary_score'] for r in records)
    close = [r for r in records if best - r['primary_score'] <= .005 + 1e-12]
    return min(close, key=lambda r: (r['inference_ms_per_text'], r['full_model_size_bytes'], r['model']))['model']
