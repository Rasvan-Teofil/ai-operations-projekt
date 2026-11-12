# Japanisch / Japanese / 日本語

Japanisch gehört zum aktuellen Projektstand auf **main**. Anwendung, Datenherkunft,
Messungen und Tests sind auf demselben Hauptzweig vorhanden.

## Aufgabe und Daten

Native japanische Nachrichten-Vollartikel werden mit derselben Rubrik als
`negative`, `neutral` oder `positive` klassifiziert. Es geht um die berichtete
Entwicklung für das Hauptsubjekt, nicht um eine emotionale Schreibweise.
Die japanischen Testtexte sind keine Übersetzungen deutscher oder englischer Artikel.

- **Train: 91** japanische Artikel; Gesamttrain 240 (DE78/EN71/JA91).
- **Test: 300** japanische Artikel; gemeinsamer Gesamttest 691 (DE273/EN118/JA300).
- Japanischer Test: 167 negative, 65 neutrale, 68 positive Referenzen.
- Quelle: japanisches Wikinews, feste Seitenrevisionen, Text-Hashes und
  beitragsbezogene Rechte-/Attributionsnachweise.
- Referenzen: **AI-labelled · Codex-reviewed**. Zwei Lesungen desselben
  AI-Annotators, keine unabhängig menschlich validierte Ground Truth.

## Gemessene Ergebnisse

| Methode | Japanisch Macro-F1 |
|---|---:|
| Frozen MiniLM + trainierter Kopf | 0.6257 |
| Eigenes MiniLM-Fine-Tuning | 0.5655 |
| Qwen mit Nachrichtenrubrik | 0.8143 |
| Eigenes Qwen LoRA, vor Test gewählter Standard | 0.8132 |

Qwen LoRA wurde anhand der Train-internen DE/EN-Hauptkennzahl gewählt. Auf
Japanisch liegt Zero-Shot geringfügig darüber; wir behaupten keinen LoRA-Gewinn
für jede Sprache. JA ist eine separat berichtete Erweiterung, nicht Bestandteil
der vorab festgelegten DE/EN-Hauptkennzahl. Keine erneute Testinferenz und keine
nachträgliche Wahl anhand dieser Tabelle.

## Anwendung und Grenzen

Japanischen Volltext in das Textfeld der lokalen Anwendung einfügen oder genau
eine öffentlich abrufbare Artikel-URL eingeben. Keine zusätzliche Sprachwahl
und keine vorgeschaltete Übersetzung. Ausgabe und API-Klassen bleiben
`negative`, `neutral`, `positive`; die Oberfläche ist bislang englisch.

Für die Vorführung gibt es den Button **Negative · JA / 日本語**. Er setzt einen
selbst formulierten synthetischen Kontrollsatz ein, keinen Testartikel:

> 強い地震で多数の住宅が倒壊し、住民が負傷した。

Sinngemäß: Bei einem starken Erdbeben stürzten zahlreiche Häuser ein, Bewohner
wurden verletzt. Dieser Satz gehört weder zum Train noch zum gemessenen Test.

Maximal 20000 Zeichen, bei Qwen höchstens 4096 Artikeltokens. Zu lange Eingaben
werden abgewiesen. Japanische Satzgrenzen und Originaltext-Offsets wurden mit
synthetischen Eingaben geprüft. Die gemessene Artikelqualität ist kein Nachweis
für die Genauigkeit jeder einzelnen Satzmarkierung.

Das Testarchiv ist eingeschränkt; Rechteausschlüsse, Längengrenzen, mögliche
Vortrainingsüberschneidungen und die AI-Referenzen begrenzen die Übertragbarkeit.
Öffentliches Hosting ohne Tower ist noch nicht eingerichtet.

## Nachweise auf main

- [Vollständiger Ergebnisbericht](../../reports/next_stage/comparison-v1.md)
- [Modellkarten](../../reports/next_stage/model-cards-v1.md)
- [Testmanifest mit Sprach-/Klassenhäufigkeiten](../../reports/next_stage/final-test-v1/test-manifest.json)
- [Quellenherkunft](../../reports/next_stage/final-test-v1/provenance.jsonl)
- [Lizenzinventar](../../licenses/inventory.json)
- [GPU-/HTTP-Prüfungen einschließlich JA und Unicode-Offsets](../../reports/next_stage/live-verification-v1.json)
- [Technische Prüfungen](../../reports/next_stage/verification-final-v1.json)
- [Reproduktion](../reproduce-full-article.md)
