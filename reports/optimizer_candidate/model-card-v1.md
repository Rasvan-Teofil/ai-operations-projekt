# Modellkarte: article-qwen-lora-optimizer-v1

| Eigenschaft | Wert |
|---|---|
| Rolle | Separater explorativer Kandidat; bisheriges API-Modell bleibt Standard |
| Basis | Qwen/Qwen3-4B-Instruct-2507, Revision cdbee75f17c01a7cc42f958dc650907174af0554 |
| Aufgabe | Vollständige deutsche, englische und japanische Nachrichten: negative / neutral / positive |
| Daten | Bestehende 240 Train-Artikel: DE 78, EN 71, JA 91 |
| Klassenreferenzen | Projektseitig am 14.11.2026 geprüft; ursprüngliche Annotationsherkunft bleibt erhalten |
| LoRA | Rank 8, Alpha 16, Dropout 0,05; q_proj und v_proj; frische Initialisierung |
| Optimierung | AdamW, Lernrate 0,00005, Weight Decay 0, Clipping 1,0, kein Warmup |
| Training | Drei Epochen, Akkumulation 16 Artikel, 45 Schritte, Seed 20261110 |
| Loss | Artikelweise A/B/C-Cross-Entropy; inverse Klassengewichte ausschließlich aus allen 240 Train-Artikeln |
| Präzision | BF16-Basis, FP32-Adapter und AdamW-Zustände |
| Kontext | Höchstens 4096 Artikeltokens und 4608 Tokens mit Prompt; kein stilles Abschneiden |
| Auswahl des Checkpoints | Fester letzter Schritt; keine Güteprüfung oder Auswahl während des Fits |
| Adaptergröße | 2.949.120 Parameter; rund 11,3 MiB einschließlich Adapter-Metadaten |
| Unabhängiger Qualitätswert | Noch nicht erhoben |
| Endgültiger Test | Keine erneute Verwendung; bisherige Werte gelten ausschließlich für das alte Modell |
| API-Alias | article_qwen_lora_optimizer_v1 |
| Lokales Inferenzprofil | models/optimizer_candidate/v1/profile |
| Lokaler Adapter | models/optimizer_candidate/v1/adapter |

Die vier Parameter stammen aus der [abgeschlossenen Ablation](../optimizer_ablation/interpretation-v1.md).
Auswahl nach dem Mittel beider Seeds, bei Gleichstand geringerer Weight Decay.
Der neue Seed ist vor dem Fit fest 20261110; kein gezieltes Übernehmen des besseren
Einzel-Seeds. Diese zusätzliche Auswahlregel ist eine nachträgliche explorative
Entscheidung und wird nicht als Teil der ursprünglichen Bestätigungsstudie dargestellt.

Die 81 Artikel, die zuvor Entwicklung waren, gehören jetzt zum Training. Die
früheren Entwicklungswerte dürfen deshalb weder als unabhängige Güte dieses
Artefakts übernommen noch durch eine erneute Auswertung derselben Fälle bestätigt
werden. Technische Tests mit wenigen synthetischen Texten prüfen Funktionsfähigkeit,
keine Klassifikationsgüte oder Kalibrierung. Kein belegter Qualitätsgewinn gegenüber
dem bisherigen API-Standard. Für weitere Qualitätsprüfung sind zusätzliche
Train-interne Untersuchungen oder eine später unabhängig geplante frische Evaluation nötig.

Eingabevorgaben, Rubrik, Labelreihenfolge und gemeinsamer Inferenzlader entsprechen
der eingefrorenen Vollartikel-Anwendung. Drei Scores entstehen durch Softmax der
A/B/C-Logits; sie sind keine kalibrierten Wahrscheinlichkeiten. Satzfarben in der
Anwendung sind zusätzliche Vorhersagen und keine kausale Erklärung der Artikelklasse.

Gewichte bleiben lokal und teilen sich dieselben Basisgewichte; keine zusätzliche
Basismodellkopie. Veröffentlichte Berichte enthalten Methoden, Hashes, Telemetrie,
synthetische Beispiele und Laufkennungen, keine Artikelrohtexte. Quellen- und
Datenrechte bleiben im [Quellenkatalog](../../docs/data/README.md) nachvollziehbar.
