"""Tests bleiben auf der Dummy-Stichprobe. Kein Download, kein Transformer."""

import os

os.environ.setdefault("SENTIMENT_DATA", "dummy")
