# Python-Umgebungen

`../requirements.txt` ist der gemeinsame Einstieg. Die fünf ursprünglichen
Zusatzdateien wurden ohne Byteänderung hierher verschoben:

- `requirements-cuda.txt`: separate CUDA-Torch-Installation auf dem Tower.
- `requirements-evaluation.txt`: Übersetzungs-/Benchmarkzusätze, ohne Torch.
- `requirements-local-llm.txt`: lokale LLM-/Berichtsumgebung, schließt Evaluation ein.
- `requirements-llm-adaptation.txt`: PEFT, Accelerate und Ressourcenmessung.
- `requirements-lock.txt`: aufgezeichnete Tower-Umgebung; kein plattformunabhängiger Lock.

Vom Repository-Wurzelverzeichnis:

```bash
python -m pip install -r requirements/requirements-cuda.txt
python -m pip install -r requirements.txt -r requirements/requirements-local-llm.txt -r requirements/requirements-llm-adaptation.txt
```

Eine bestehende funktionierende Umgebung nicht routinemäßig neu installieren.
CI verwendet die CPU-Wheel-Variante separat; die ersetzt nicht die Tower-Umgebung.
Alte Archivbefehle nennen zum Teil noch die damaligen Wurzelpfade.
