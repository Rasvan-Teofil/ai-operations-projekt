"""Preserve committed historical reports when reproducing presentation output."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ('development', 'local_llm', 'common_data_v3', 'expanded_news_v4', 'llm_adaptation_v5')


def report_destination(folder):
    folder = Path(folder).resolve()
    for name in HISTORICAL:
        original = (ROOT / 'reports' / name).resolve()
        if folder == original or folder.is_relative_to(original):
            raise RuntimeError('Historical reports are preserved. Use the documented reproduction output paths; do not render over original evidence.')
    return folder
