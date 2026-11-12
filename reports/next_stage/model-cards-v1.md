# Vollartikel-Modellkarten v1

Alle vier Kandidaten verwenden dieselben 240 Train-Artikel. Die Bedingungen wurden vor Erhebung des frischen Tests eingefroren; **Qwen LoRA wurde auf der Entwicklung gewählt**. Seine Wahl bleibt nach Test unverändert. Neue MiniLM-Modelle starten vom sauberen Encoder, nicht von früheren Finanz-Fine-Tunes.

## Daten und Messung

Train: DE78/EN71/JA91. Test: DE273/EN118/JA300, 691 Artikel in 658 beobachteten Ereignis-/Quellengruppen. Native Wikinews-Volltexte, begrenzte direkte Zitate, maximal 20000 Zeichen und 4096 Qwen-Artikeltokens. 856 frische Rohfälle wurden vollständig zweimal gelesen, Rechte und Quellengruppen geprüft. Rohfälle bleiben Test-reserviert, auch bei Ausschluss.

Referenzen: **AI-labelled · Codex-reviewed**. Ein AI-Annotator, zwei Eigenprüfungen; keine unabhängige menschliche Validierung, genaue Serving-Revision des Annotators unbekannt. 41 Klassenabweichungen vor Inferenz begründet. Mehrdeutigkeit bleibt in der Sensitivitätsauswertung erhalten.

Primärmaß: Mittel aus DE- und EN-macro-F1, jede Sprache mit Gewicht 1/2. JA separat. Je Modell/Artikel eine gespeicherte Testvorhersage; 20 Warmups nur mit Train. Batch 1, RTX 5060 Ti, CUDA BF16; Zeiten enthalten Tokenisierung, Transfer und vollständige Aggregation, ohne Laden und URL-Abruf.

| Modell | DE/EN F1 | DE | EN | JA | ms/Artikel | vollständige MiB | Train-Sekunden |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen128 | 0.5150 | 0.5281 | 0.5019 | 0.6257 | 3.72 | 240.73 | 1.23 |
| minilm128 | 0.5928 | 0.4734 | 0.7122 | 0.5655 | 3.66 | 240.73 | 12.51 |
| qwen-zero | 0.7796 | 0.7508 | 0.8085 | 0.8143 | 172.33 | 7687.49 | 0.00 |
| qwen-lora | 0.8218 | 0.7612 | 0.8825 | 0.8132 | 184.26 | 7698.77 | 380.33 |

## Anpassungen und Grenzen

- Frozen MiniLM: eingefrorener multilingualer L12-Encoder; trainierter linearer Artikelkopf. Maskiertes Tokenmittel je Chunk, gleiches Mittel aller Chunk-Features vor Klassifikation.
- MiniLM: derselbe Kopf und Encoder vollständig angepasst, drei Epochen, Lernrate 2e-5, Akkumulation 16; 128-Token-Chunks.
- Qwen Zero: gepinnte Qwen3-4B-Instruct-2507-Basis, vollständige Nachrichtenrubrik und eingeschränkte A/B/C-Next-Token-Logits. Kein Training.
- Qwen LoRA: dieselbe Basis/Rubrik, r8/alpha16/dropout0.05 auf q/v, drei Epochen, Lernrate 1e-4, Akkumulation 16. Adapter separat, vollständige Größenangabe einschließlich Basis.

Qwen weist über 4096 Artikeltokens zurück und schneidet nichts aus der Mitte. MiniLM verarbeitet sämtliche Chunks. Historische Profile behalten ihre alten Regeln. Die Menge 240 ist durch diesen Quellenpilot begrenzt; die Train-Mengenkurve war nicht gesättigt, somit kein behauptetes Datenoptimum. 3000 war eine Obergrenze, 6000 bleibt eine spätere Forschungsfrage.

Das angepasste MiniLM ist schnell, weist aber deutliche Sprachunterschiede auf; seine deutsche F1 liegt unter der Frozen-Baseline. Die aggregierte Verbesserung beweist daher keine Verbesserung in jeder Sprache. Qwen LoRA liegt auf JA geringfügig unter Zero-Shot; kein universeller Gewinn wird behauptet.

Bootstrapintervalle und gepaarte Differenzen: [vollständiger Bericht](comparison-v1.md). Sie erfassen Stichprobenunsicherheit innerhalb dieser Population, keine Referenzfehler. Öffentliche Vortrainingsüberschneidungen und übersehene Ereignisverbindungen bleiben möglich. Scores sind nicht kalibriert; Satzfarben sind zusätzliche, separat nicht evaluierte Vorhersagen. Kein Nachweis von 95% Qualität, kein allgemeines Verlags- oder Produktionstauglichkeitsversprechen.

## Herkunft und Rechte

Qwen-Basis: Apache-2.0, Revision `cdbee75f17c01a7cc42f958dc650907174af0554`. MiniLM-Basis: Apache-2.0, Revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`. Quelltexte behalten beitrags-/revisionsbezogene CC-BY-Bedingungen, neue Annotationsmetadaten CC0 soweit Projektrechte reichen. Bild-/Zitatrechte und Modell-/Softwarelizenzen sind getrennt; offene Dritttextrechte wurden ausgeschlossen. Keine pauschale Freigabe weiterer Eingaben oder eigener Gewichte.

Exakte Artefakt-Hashes und Fit-Laufkennungen: `final-artifacts-v1.json`. Test-Laufkennungen, Klassenwerte, Konfusionsmatrizen, Trainingsspeicher und Konfidenzintervalle: `comparison-v1.json`. Model Cards liegen bewusst außerhalb der eingefrorenen Modellordner.

[Reproduktion](../../docs/reproduce-full-article.md) · [Lizenzinventar](../../licenses/inventory.json) · [Quellennachweise](final-test-v1/provenance.jsonl) · [Test-Manifest](final-test-v1/test-manifest.json)
